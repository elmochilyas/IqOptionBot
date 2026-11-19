# Phase 33: Operator-Supervised Acquisition Timing Measurement Protocol

**Status:** SPECIFICATION & OPERATOR GUIDE — AWAITING INDEPENDENT AUTHORIZATION  
**Research Standing:** NO_DEMONSTRATED_EDGE  
**Operational Mode:** STRICTLY READ-ONLY PRACTICE BROKER MEASUREMENT (ZERO ORDERS)  

---

## 1. Scope, Purpose, and Safety Boundaries

This protocol governs a small, independently authorized foreground measurement trial designed to measure empirical broker delivery and client validation latency for EURUSD 1-minute market candles on IQ Option.

### Mandatory Operational Boundaries

- **Instrument:** EURUSD only (`symbol="EURUSD"`, `candle_size_seconds=60`).
- **Maximum Candle Cycles:** Exactly 2 distinct completed candles (`--max-candles 2`).
- **Maximum Duration:** Exactly 180 seconds (3 minutes) ceiling (`--duration 180`).
- **Zero Orders:** No real, demo, practice, or synthetic trading orders are permitted or implemented.
- **PRACTICE Wallet Only:** Authenticates exclusively against a Practice wallet (`is_practice=True`). Automated wallet switching is prohibited; non-practice active accounts trigger immediate session abort.
- **Credential Hygiene:** Interactive foreground prompt only (`getpass`). Credentials are never saved to disk, `.env`, shell history, or version control.
- **No Unattended Execution:** Requires active operator presence throughout the 3-minute window.
- **Fail-Closed Teardown:** Zero reconnect loops; any transport error or timeout immediately disconnects and shuts down.
- **Immutable Evidence:** Local storage only (`data/phase33/`); files are create-only and cannot be overwritten.
- **No Automatic Follow-up Runs:** The trial terminates upon completion or cancellation and does not schedule recurring jobs.

---

## 2. Step 1 — Check Prerequisites

Before starting the supervised measurement:

1. **Verify Local Clock Stability & Discipline:**
   Confirm local OS system clock is disciplined via NTP. On Windows:
   ```powershell
   w32tm /query /status
   ```
   On Linux/macOS:
   ```bash
   timedatectl status
   ```

2. **Verify Offline Code Quality & Security Baseline:**
   From the repository root (`C:\Users\elmoc\Desktop\IqOptionBot`), run:
   ```powershell
   .venv\Scripts\python.exe -m iqoption_research.cli.main security-check
   .venv\Scripts\python.exe -m ruff check src tests
   .venv\Scripts\python.exe -m mypy src
   ```
   All checks must pass with zero errors.

3. **Verify Timing Replay Harness:**
   Execute all 10 offline deterministic simulation scenarios:
   ```powershell
   .venv\Scripts\python.exe -m iqoption_research.cli.main timing replay
   ```
   Confirm all scenarios pass safety invariants.

4. **Verify Target Evidence Directory:**
   Ensure the destination directory exists:
   ```powershell
   New-Item -ItemType Directory -Force -Path data/phase33
   ```

---

## 3. Step 2 — Start the Measurement

When authorized, execute the measurement command in an interactive terminal session:

```powershell
.venv\Scripts\python.exe -m iqoption_research.cli.main timing measure `
  --live `
  --yes-i-understand `
  --max-candles 2 `
  --duration 180
```

### CLI Flag Explanations

- `--live`: Enables the read-only live broker measurement facade.
- `--yes-i-understand`: Explicit confirmation flag proving the operator is present and acknowledges the operational bounds.
- `--max-candles 2`: Hard ceiling of two observed 1-minute candle boundaries.
- `--duration 180`: Hard session timeout of 180 seconds (3 minutes).
- `--overwrite`: **Forbidden.** In live mode, attempting to pass `--overwrite` results in immediate command refusal.

---

## 4. Step 3 — Provide Credentials Locally

1. When prompted by the CLI:
   ```text
   Enter IQ Option Practice credentials (interactive prompt only):
   Broker Account Email: operator@example.com
   Broker Account Password: [hidden]
   ```
2. Type the account email and press Enter.
3. Type or paste the account password. Password input is handled via `getpass` and characters will not echo on screen.
4. **Agent Boundary:** Never provide credentials to an AI assistant, chat interface, prompt, or repository file.

---

## 5. Step 4 — Supervise and Stop the Measurement

The measurement executes in the foreground, displaying stage checkpoints in real time:

```text
[1/4] Verifying broker account and wallet authority...
      PRACTICE wallet verified successfully.
[2/4] Initializing clock discipline anchor and stability checks...
      Anchor UTC epoch ns: 1795000000000000000
      NTP uncertainty budget: +/-25.00 ms

[3/4] Starting observation window (max 180s)...

      Observing F=1795063200 (now=1795063202, offset=2s)...
      Candle F=1795063200 observed:
        - Client frame receipt:     2341.12 ms from candle close [compliant: YES (within 15s)]
        - Candle validation:        2356.45 ms from candle close [compliant: YES (within 15s)]
        - Full prediction contract: NOT_MEASURED (timing measurement only)
        - Observation verdict:      ACQUISITION_15S_COMPLIANT
```

### Session Termination

- **Normal Completion:** The session terminates automatically after 2 candle cycles or 180 seconds.
- **Graceful Manual Interruption:** The operator may abort at any time by pressing `Ctrl+C`. The session catches the interrupt, disconnects cleanly from the WebSocket, and persists whatever completed observations were gathered to that point.
- **Connection Interruption:** If the broker disconnects or times out, the session fails closed immediately without reconnect attempts.

---

## 6. Step 5 — Verify the Evidence File

Upon completion, the session persists an immutable, create-only JSON registry at:
`data/phase33/supervised_timing_evidence_meas_<session_id>.json`

Verify the evidence file using the Phase 33 verification tool:

```powershell
.venv\Scripts\python.exe -m iqoption_research.cli.main timing verify-evidence `
  --path data/phase33/supervised_timing_evidence_meas_<session_id>.json
```

### Verification Checks Performed

- SHA-256 integrity hash verification over all observations and the registry envelope.
- Observation sequential index integrity (no missing, skipped, or reordered items).
- Candle boundary uniqueness and strict chronological increase.
- Strict provenance separation (rejection of any synthetic data in live evidence).
- Summary count reconciliation against actual observation states.
- Rejection of any acquisition-only observation claiming Phase 30 satisfaction.

---

## 7. Step 6 — Interpret Receipt versus Validation Latency

The evidence registry separates three distinct conclusions:

1. **Client Frame Receipt Latency (`broker_delivery_offset_ms`):**
   $$\Delta t_{\text{recv}} = t_{\text{socket\_recv}} - F$$
   The empirical latency from nominal candle close $F$ until the broker's WebSocket message was read off the local socket.

2. **Candle Validation Latency:**
   $$\Delta t_{\text{val}} = t_{\text{val}} - F$$
   The empirical latency until payload JSON decoding, candle normalization, data quality validation, and instrument checks completed.

3. **Prediction Contract Status:**
   Always reported as `NOT_MEASURED` (`phase30_gate_satisfied=False`, `decision="NOT_MEASURED"`). Acquisition timing measurement does not execute features, inference, or journal commits. An acquisition observation must **never** be cited as evidence of forecast feasibility or trading viability.

---

## 8. Step 7 — Report Errors Without Exposing Sensitive Data

If an error or network drop occurs:

- The evidence file is sanitized by default (`sanitize_payload`), redacting passwords, tokens, balances, cookies, and user IDs.
- When sharing diagnostic output or submitting reports, share only:
  - Session ID.
  - Latency statistics (median delivery offset, validation offset).
  - Verdict labels (`ACQUISITION_15S_COMPLIANT`, `FAIL_GATE_OVERRUN`, `FAIL_CLOCK_STEP_DETECTED`).
  - Python stack traces with sensitive arguments omitted.
- Never paste raw authorization handshakes or credential prompts into issues, PRs, or public channels.
