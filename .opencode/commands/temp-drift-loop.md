---
description: Run or resume the full TEMP-DRIFT research loop with visible output
agent: build
subtask: false
---

Run or resume the TEMP-DRIFT benchmark workflow directly in this OpenCode TUI session.

First read these files completely:

- `AGENTS.md` (if it is missing, copy `ResearchLoop/AGENTS.template.md` to `AGENTS.md`)
- `ResearchLoop/config.json`
- `ResearchLoop/phases.json`
- `ResearchLoop/phase_contracts.md`
- `ResearchLoop/reference/paper_protocol.md`
- `Reports/status.json` if it exists

Then execute the autonomous engineering loop from the earliest phase that has not passed its independent gate. Do real repository inspection, implementation, tests, experiments, audits, statistics, plots, and reports. Use only the exact Python interpreter from `ResearchLoop/config.json` for Python commands.

Important TUI behavior:

1. Work directly with tools in this current session. Do NOT run `ResearchLoop/RUN_LOOP.ps1` and do NOT launch another `opencode` process.
2. Before each substantial command, print one short line explaining what will run.
3. Let command stdout/stderr remain visible in the TUI. Also preserve full logs under `Reports/logs/` when useful.
4. After each repair attempt, show the actual PASS/FAIL evidence concisely.
5. Update `Reports/status.json` atomically after every meaningful unit so running `/temp-drift-loop` again resumes instead of restarting.
6. Never rerun a completed valid experiment. Detect per-run completion markers and hashes first.
7. Never fabricate metrics, copy paper values into our results, weaken audits, or mark a failed phase PASS.
8. When a phase appears complete, run its independent gate with the configured Python interpreter:
   `ResearchLoop/tools/gate.py <project-root> ResearchLoop/config.json <phase-id>`.
9. If the gate fails, diagnose, fix, rerun, and show the failure and new evidence in this TUI.
10. Continue automatically to the next phase only after PASS. Stop only for a true scientific blocker defined in `AGENTS.md`, and write `Reports/diagnostic_blocker.md` first.

Optional command arguments supplied by the user: `$ARGUMENTS`

If arguments name a phase, treat that as a request to start/resume there only when all prerequisite phases are already PASS. Otherwise resume normally from the earliest incomplete phase.

Begin now. Start with a concise status summary, then run the first required command instead of only presenting a plan.

