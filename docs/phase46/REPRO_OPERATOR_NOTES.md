# Phase 46 — Operator Instructions, Synthetic Demo & Reproduction Notes

**Governance:** `execution_authority = NONE`, `NO_DEMONSTRATED_EDGE`, `DEVELOPMENT_RESEARCH_NOT_INDEPENDENT_CONFIRMATION`.  
No broker connections during development or automated testing; no real/demo orders; no wallet mutations.

---

## 1. Launching the Desktop Research Workstation

```powershell
.\Start-ResearchDashboard.cmd
```

Navigate to **Shadow Evaluation → Prospective Observation** to open the **Phase 46 Observation Center**.

## 2. Offline Synthetic Demonstration Instructions (Operator)

1. Ensure **Observation Mode** is set to **Offline Synthetic Demonstration (Phase 46)**.
   - Note: Switching to **Genuine Observation (Blocked — Unauthorized)** displays an explicit blocked status and disables **Start Observation**, because 0 candidates qualified in WS2 and `docs/phase46/PREREGISTERED_PROSPECTIVE_CONTRACT.md` remains `DRAFT / UNAUTHORIZED`.
2. Select a synthetic fixture in **Model / Fixture** (`SYNTH_EURUSD_M1_H1_FIXTURE`, `SYNTH_EURUSD_M5_H3_FIXTURE`, or `SYNTH_EURUSD_M15_H2_FIXTURE`) or choose the native **Timeframe** (`M1 (60s)`, `M5 (300s)`, `M15 (900s)`). The **Prediction Duration** updates automatically (`1 min (H1)`, `15 min (H3)`, `30 min (H2)`).
3. Choose **Duration (min)** and **Sim Pace** (`Interactive (120ms/bar)`, `Fast (25ms/bar)`, or `Instant (0ms/bar)`).
4. Click **Start Observation**.
   - A background worker provisions the isolated synthetic workspace under `data/phase46/synthetic_workspace/` and streams closed synthetic candles through `ProspectiveShadowObserver`.
   - Live KPIs (opportunities, selected, pending, resolved W/L/T, abstentions, accuracy, coverage, matched baseline, hypothetical 85% payout sensitivity) and the **Cumulative Accuracy (Synthetic)** chart update via a thread-safe UI queue.
5. Click **Stop / Cancel** at any time to cooperatively cancel and seal partial evidence, or wait for `STREAM_COMPLETED_EOF`.
6. In **Observation Session History & Sealed Evidence**:
   - Select any session (labeled `SYNTHETIC` or `GENUINE`).
   - Click **Open Diagnostics Report** to compute WS6 diagnostics.
   - Click **Verify Session Integrity** to run cryptographic sealing verification.
   - Click **Export Sealed Evidence** to write verified JSON/Markdown diagnostics, session files, and `verification_result.json` to `data/phase46/exports/<session_id>/`.

### Captured Desktop Screenshots
- `docs/phase46/screenshots/01_observation_center_synthetic_ready.png`
- `docs/phase46/screenshots/02_observation_center_genuine_blocked.png`
- `docs/phase46/screenshots/03_observation_center_synthetic_completed.png`

---

## 3. Reproduction Commands

### WS1 Focused Scientific Validation (10,000-replicate day-block + 2-day sensitivity + nested)
```powershell
.\.venv\Scripts\python.exe scripts\run_phase46_research.py focused --bootstrap-replicates 10000
```

### Full Repository Validation
```powershell
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m ruff format src tests --check
.\.venv\Scripts\python.exe -m mypy src
.\.venv\Scripts\python.exe -m iqoption_research.cli.main security-check --json
.\.venv\Scripts\python.exe scripts\ci_pytest.py
```

---

## 4. Remaining Limitations
- All three Phase 45 focused candidates remain `NO_DEMONSTRATED_EDGE` with 10k day-block 95% CI lower bounds below the hypothetical 54.05% break-even threshold and low coverage (1.48%–2.48%).
- No candidates qualify for freezing under preserved Phase 40C / Stage C acceptance gates.
- Genuine prospective observation remains strictly blocked until an eligible candidate is frozen and the preregistered contract is finalized and owner-authorized.
