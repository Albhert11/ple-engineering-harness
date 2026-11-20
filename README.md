# PLE Engineering Agent Harness (`ple-engineering-harness`)

A multi-stage Plan-Work-Audit-Verify agent harness built for **Physical-Engineering's Last Exam (PLE)** using `google-cloud/gemini-4-argon`.

## Architecture
1. **Automated Workspace Inventory (`scan_workspace`)**: Maps directory hierarchy, file extensions (`.ifc`, `.pdf`, `.xlsx`, `.csv`, `.py`), and byte sizes before planning to save discovery turns.
2. **Principal Engineering Planner (`PLANNER`)**: Read-only stage (`10%` time budget) identifying deliverable paths, governing engineering standards (ACI, AISC, Eurocode, IEC, NEC, ASHRAE, ASME), unit-conversion traps, and computational workflows.
3. **Multidisciplinary Engineering Worker (`WORKER`)**: Enforces Python-first numerical calculation, explicit unit tracking, and solver residual verification before writing deliverables.
4. **Deterministic Pre-Verifier Audit (`audit_deliverables`)**: Non-destructive check flagging empty files (`0 bytes`) or unresolved tokens (`TODO`, `TBD`, `FIXME`, `NaN`) before invoking the QA/QC Verifier.
5. **Independent QA/QC Verifier (`VERIFIER`)**: Fresh-context audit (`up to 2 fix rounds`) recomputing key engineering numbers via Python and verifying deliverable schemas.
