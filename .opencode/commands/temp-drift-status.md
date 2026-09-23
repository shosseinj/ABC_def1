---
description: Show TEMP-DRIFT phase, latest gate, logs, and blockers
agent: build
subtask: false
---

Inspect `Reports/status.json`, `Reports/receipts/`, `Reports/logs/`, and `Reports/diagnostic_blocker.md` if present. Show:

- current overall status and phase;
- PASS/FAIL state for every phase;
- current or latest run and its evidence;
- the last 30 relevant log lines without exposing secrets;
- any exact blocker and the next safe action.

Do not modify experiments, results, reports, or status. This is a read-only status command.

