# Phase 46 — Repro & Operator Notes

## Repro (WS1 focused validation)
From repo root:

```bash
python scripts/run_phase46_research.py focused --repo-root . --bootstrap-replicates 10000
```

Artifacts (sanitized): `reports/phase46/focused_validation.json`, `reports/phase46/focused_validation.md`.

## Operator notes (supervised observation intent)
- Use existing desktop observation/shadow UI paths; **do not** enable unattended/auto-start/auto-reconnect behavior.
- Select saved account only through existing authenticated flows; **no broker calls in automated tests**.
- Start/Stop explicitly. Show measured values only. Label everything as DEVELOPMENT/PROSPECTIVE per contract.
- No orders, no wallet mutations, no auto model promotion.

## Governance
- `execution_authority = NONE`, `NO_DEMONSTRATED_EDGE`.
- Preserve Phase 40C/Stage C contracts; any change requires explicit justification + owner authorization.