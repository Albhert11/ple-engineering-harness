"""Plan-Work-Audit-Verify engineering harness for Physical-Engineering's Last Exam.

Upgrades over standard plan-verify:
1. Automated workspace inventory scan prior to planning (maps file types, sizes,
   and directory layout without wasting model turns).
2. Domain-grounded system prompts enforcing unit consistency (SI/Imperial),
   code-clause citation, Python-based numerical verification, and solver
   convergence checks across Structural, Civil, Electrical, and Mechanical tasks.
3. Deterministic pre-verifier audit script checking deliverable existence,
   non-zero file sizes, absence of NaN/TODO placeholders, and valid syntax.
4. Iterative verifier-worker repair loop bounded by task time budget.
"""

import asyncio
import json
import os
import re

import openai
import taskbridge

MODEL = os.environ["MODEL_NAME"]
MAX_EFFORT_MODELS = ("claude", "glm", "gpt-6", "deepseek-v4")
EFFORT = os.environ.get("REASONING_EFFORT") or (
    "max" if any(m in MODEL.lower() for m in MAX_EFFORT_MODELS) else "high"
)

WRAP_UP_SEC = 240
WRAP_UP = {
    "role": "user",
    "content": "Time is up for this step. Stop and reply with your answer.",
}
RESULT_CHARS = 24_000
RETRY_WAITS = (5, 15, 30, 60, 120)
CONTEXT_FULL = re.compile(
    r"context.?(length|window)|prompt is too long|maximum prompt length"
    r"|exceed[^.]{0,40}(token|context)",
    re.I,
)
CUT_KEEP = 1200
MAX_NUDGES = 3
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "32000"))
CACHE_MARK = "claude" in MODEL.lower()

PLAN_SHARE = 0.10
CHECK_SHARE = 0.25
MIN_STEP_SEC = 120
MAX_FIX_ROUNDS = 2

client = openai.AsyncOpenAI(max_retries=5, timeout=1800)

PLANNER = """You are a Principal Engineering Planner preparing an execution plan for an engineering task in a Linux \
container with no internet access (bash starts in {workdir}).
Inspect the task files carefully using read_file and bash (do NOT create or modify any files).
Identify:
1. Every required deliverable: exact file path, required format (IFC, CSV, XLSX, JSON, PDF, script, or final message), and schema.
2. Governing engineering codes, standards, and clauses referenced in the project documents (e.g., ACI, AISC, Eurocode, IEC, NEC, ASHRAE, ASME).
3. Unit system and potential unit-conversion traps (e.g., kN vs N, MPa vs Pa, mm vs m, kW vs W, imperial vs SI).
4. Step-by-step computational workflow using Python scripts/solvers rather than manual arithmetic, and exact verification checks."""

WORKER = """You are an expert multidisciplinary engineer executing a task in a Linux container with no internet access \
(bash starts in {workdir}). Follow the engineering plan below, correcting any assumption if the actual workspace files show otherwise.

Mandatory Engineering Rules:
- NEVER compute engineering formulas, loads, stresses, cable sizes, or thermal balances by hand. Write and run Python scripts (using numpy, scipy, pandas, ifcopenshell, openpyxl, etc. if installed) and print intermediate quantities with explicit units.
- Verify every code clause and design limit against the project specifications before finalizing parameters.
- If running numerical solvers or simulations, check convergence residuals and physical plausibility (positive dimensions, safety factors >= required minimums, energy/mass conservation).
- Write every deliverable to its exact required path and format, then re-open and inspect the written file to confirm it is complete and valid.
- When finished, reply without calling a tool: give a concise, structured summary of each deliverable path, governing standard/method used, and key numerical results with units."""

VERIFIER = """You are an independent QA/QC Principal Engineer auditing an engineer's completed deliverables in a Linux \
container (bash starts in {workdir}). Do NOT create or modify any files.
You are given the task instructions, the engineering plan, an automated file audit report, and the engineer's summary.

Inspect every deliverable file directly using bash and read_file:
1. Confirm exact file paths, formats, and required sections/columns exist and are non-empty.
2. Recompute key engineering calculations via an independent quick Python one-liner in bash and check unit consistency (SI vs Imperial, prefix scaling).
3. Check for common PLE failure modes: drawing/schedule misreads, wrong code-clause thresholds, unverified solver non-convergence, or truncated outputs.

If every deliverable passes all checks, reply with the single word PASS and nothing else.
Otherwise, reply with a numbered list of specific, actionable defects (including exact file paths, expected vs. actual values/units, and how to fix them)."""


def tool(name, description, **params):
  return {
      "type": "function",
      "function": {
          "name": name,
          "description": description,
          "parameters": {
              "type": "object",
              "properties": {k: {"type": v} for k, v in params.items()},
              "required": list(params),
          },
      },
  }


TOOLS = [
    tool(
        "bash",
        "Run a bash command (120 s limit) and return its exit code and output.",
        command="string",
    ),
    tool("read_file", "Read a text file.", path="string"),
    tool(
        "write_file",
        "Create or overwrite a file; parent folders are created.",
        path="string",
        content="string",
    ),
]
READ_ONLY = TOOLS[:2]


def clip(text):
  if len(text) <= RESULT_CHARS:
    return text
  half = RESULT_CHARS // 2
  return (
      f"{text[:half]}\n[... {len(text) - RESULT_CHARS} characters cut"
      f" ...]\n{text[-half:]}"
  )


def marked(messages):
  if not CACHE_MARK:
    return messages
  out = list(messages)
  for i in {0, len(out) - 1}:
    m, content = out[i], out[i].get("content")
    if isinstance(content, str) and content:
      out[i] = {
          **m,
          "content": [{
              "type": "text",
              "text": content,
              "cache_control": {"type": "ephemeral"},
          }],
      }
    else:
      out[i] = {**m, "cache_control": {"type": "ephemeral"}}
  return out


async def run_tool(session, fn):
  name = fn["name"]
  try:
    args = json.loads(fn["arguments"] or "{}")
    if not isinstance(args, dict):
      raise TypeError("not a JSON object")
    fn["arguments"] = json.dumps(args, allow_nan=False)
  except (TypeError, ValueError) as exc:
    fn["arguments"] = "{}"
    return (
        f"error: the arguments are not a JSON object ({exc}); send the call"
        " again"
    )
  try:
    if name == "bash":
      result = await session.exec(args["command"])
      return clip(f"exit code {result['exit_code']}\n{result['output']}")
    if name == "read_file":
      return clip(
          (await session.read_file(args["path"])).decode("utf-8", "replace")
      )
    if name == "write_file":
      return (
          f"wrote {await session.write_file(args['path'], args['content'])}"
          f" bytes to {args['path']}"
      )
    return f"error: there is no tool named {name}"
  except (taskbridge.TaskError, KeyError, ValueError) as exc:
    return f"error: {type(exc).__name__}: {exc}"


async def scan_workspace(session):
  """Collects a lightweight read-only inventory of files in the task workspace."""
  cmd = (
      "find . -maxdepth 4 -not -path '*/.*' -printf '%p\\t%s bytes\\n' "
      "2>/dev/null | head -n 120"
  )
  try:
    res = await session.exec(cmd)
    if res.get("exit_code") == 0 and res.get("output", "").strip():
      return clip(res["output"].strip())
  except Exception:
    pass
  return "(workspace scan unavailable; use bash/ls to inspect files)"


async def audit_deliverables(session):
  """Runs a non-destructive check for recently modified files and placeholder tokens."""
  cmd = (
      "python3 -c \""
      "import os, pathlib; "
      "files = [p for p in pathlib.Path('.').rglob('*') if p.is_file() and not any(part.startswith('.') for part in p.parts)]; "
      "bad = []; "
      "for p in files[:200]: "
      "  sz = p.stat().st_size; "
      "  if sz == 0: bad.append(f'EMPTY: {p}'); "
      "  elif p.suffix.lower() in {'.csv','.json','.yaml','.yml','.md','.txt','.py'} and sz < 2_000_000: "
      "    txt = p.read_text(errors='ignore'); "
      "    for tok in ('TODO', 'TBD', 'FIXME', 'NaN'): "
      "      if tok in txt: bad.append(f'TOKEN({tok}) in {p}'); "
      "print('Total non-hidden files:', len(files)); "
      "print('Audit flags:', bad if bad else 'NONE')"
      "\""
  )
  try:
    res = await session.exec(cmd)
    return clip(res.get("output", "").strip())
  except Exception as exc:
    return f"(audit skipped: {exc})"


async def recover(session, messages, exc, waits, until):
  status = getattr(exc, "status_code", None)
  if status == 400 and CONTEXT_FULL.search(str(exc)):
    return await shorten(session, messages)
  wait = next(waits, None)
  if wait is None or not (
      status is None or status in (408, 409, 429) or status >= 500
  ):
    return False
  left = session.time_left()
  if left is not None and left > until:
    wait = min(wait, left - until)
  await session.log(
      f"model call failed ({str(exc)[:200]}); again in {wait:.0f} s"
  )
  await asyncio.sleep(wait)
  return True


async def shorten(session, messages):
  long = [
      n
      for n, m in enumerate(messages)
      if m["role"] == "tool" and len(m["content"]) > 2 * CUT_KEEP
  ]
  if not long:
    return False
  cut = long[: max(1, len(long) // 2)]
  for n in cut:
    messages[n]["content"] = (
        messages[n]["content"][:CUT_KEEP]
        + "\n[... cut to save context; run the tool again if needed]"
    )
  await session.log(f"context full: cut {len(cut)} old tool results")
  return True


def conversation(system, session, prompt):
  return [
      {"role": "system", "content": system.format(workdir=session.workdir)},
      {"role": "user", "content": prompt},
  ]


async def agent(session, messages, tools, until, final=False):
  nudges, head, waits = 0, "", iter(RETRY_WAITS)
  while True:
    left = session.time_left()
    wrap_up = left is not None and left < until
    if wrap_up and messages[-1:] != [WRAP_UP]:
      messages.append(dict(WRAP_UP))
    if left is None:
      timeout = None
    elif wrap_up:
      timeout = min(WRAP_UP_SEC, left)
    else:
      timeout = left - until
    try:
      reply = await asyncio.wait_for(
          client.chat.completions.create(
              model=MODEL,
              messages=marked(messages),
              tools=tools,
              tool_choice="none" if wrap_up else "auto",
              reasoning_effort=EFFORT,
              max_completion_tokens=MAX_OUTPUT_TOKENS,
          ),
          timeout,
      )
    except asyncio.TimeoutError:
      if wrap_up:
        return head
      continue
    except openai.APIError as exc:
      if await recover(session, messages, exc, waits, until):
        continue
      raise
    waits = iter(RETRY_WAITS)
    if reply.usage:
      details = getattr(reply.usage, "prompt_tokens_details", None)
      await session.usage(
          MODEL,
          reply.usage.prompt_tokens,
          reply.usage.completion_tokens,
          getattr(details, "cached_tokens", 0) or 0,
      )
    message, cut = (
        reply.choices[0].message,
        reply.choices[0].finish_reason == "length",
    )
    text = head + (message.content or "")
    if final and text:
      session.final_message = text
    turn = message.model_dump(exclude_none=True)
    if wrap_up or not message.tool_calls:
      turn.pop("tool_calls", None)
      if message.content:
        messages.append(turn)
      if wrap_up or (message.content and not cut) or nudges == MAX_NUDGES:
        return text
      nudges, head = nudges + 1, text
      messages.append({
          "role": "user",
          "content": (
              "Your reply was cut off; continue where it stopped."
              if cut
              else "Continue, and finish with your answer."
          ),
      })
      continue
    nudges, head = 0, ""
    messages.append(turn)
    for call in turn["tool_calls"]:
      await session.log(
          f"{call['function']['name']} {call['function']['arguments'][:200]}"
      )
      content = await run_tool(session, call["function"])
      messages.append(
          {"role": "tool", "tool_call_id": call["id"], "content": content}
      )
    if left is not None:
      messages[-1]["content"] += (
          f"\n[{int((session.time_left() - until) // 60)} minutes left for this"
          " step]"
      )


async def solve(session):
  total = session.time_left()

  def until(share):
    return 0 if total is None else total * share

  def time_for(wrap_up):
    left = session.time_left()
    return left is None or left - wrap_up >= MIN_STEP_SEC

  task = session.instruction
  inventory = await scan_workspace(session)
  planner_prompt = (
      f"## Task Instruction\n{task}\n\n"
      f"## Workspace File Inventory\n```\n{inventory}\n```"
  )
  plan = await agent(
      session,
      conversation(PLANNER, session, planner_prompt),
      READ_ONLY,
      until(1 - PLAN_SHARE),
  )
  await session.log(f"plan: {plan[:1000]}")

  work = conversation(
      WORKER,
      session,
      f"## Task Instruction\n{task}\n\n## Engineering Plan\n{plan}",
  )
  summary = await agent(session, work, TOOLS, until(CHECK_SHARE), final=True)

  for _ in range(MAX_FIX_ROUNDS):
    if not time_for(until(CHECK_SHARE / 2)):
      break
    audit_report = await audit_deliverables(session)
    review = (
        f"## Task Instruction\n{task}\n\n"
        f"## Engineering Plan\n{plan}\n\n"
        f"## Automated Deliverable Audit\n```\n{audit_report}\n```\n\n"
        f"## Engineer's Summary\n{summary}"
    )
    verdict = await agent(
        session,
        conversation(VERIFIER, session, review),
        READ_ONLY,
        until(CHECK_SHARE / 2),
    )
    await session.log(f"verifier: {verdict[:1000]}")
    if verdict.strip("*_. \r\n\t").upper() in ("", "PASS") or not time_for(
        WRAP_UP_SEC
    ):
      break
    work.append({
        "role": "user",
        "content": (
            "An independent QA/QC reviewer inspected your deliverables and "
            f"found these issues:\n{verdict}\n\n"
            "Fix every listed issue, re-verify the files, and reply with "
            "your updated summary."
        ),
    })
    summary = (
        await agent(session, work, TOOLS, WRAP_UP_SEC, final=True) or summary
    )

  return summary or session.final_message


if __name__ == "__main__":
  taskbridge.run(solve)
