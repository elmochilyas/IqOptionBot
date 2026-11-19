# Phase 33: Operator-Supervised Acquisition Timing Measurement Protocol

**Status:** SPECIFICATION & OPERATOR GUIDE — AWAITING INDEPENDENT AUTHORIZATION  
**Research Standing:** NO_DEMONSTRATED_EDGE  
**Operational Mode:** WALLET-INDEPENDENT READ-ONLY CANDLE MEASUREMENT (ZERO ORDERS)

---

## 1. Scope, Purpose, and Safety Boundaries

This protocol governs a small, independently authorized foreground measurement trial designed to measure empirical broker delivery and client validation latency for EURUSD 1-minute market candles on IQ Option.

### Mandatory Operational Boundaries

- **Instrument:** EURUSD only (`symbol="EURUSD"`, `candle_size_seconds=60` fixed defaults).
- **Maximum Candle Cycles:** Exactly 2 distinct completed candles (`--max-candles 2`).
- **Maximum Observation Loop Duration:** Exactly 180 seconds (3 minutes) ceiling (`--duration 180`).
- **Zero Orders:** No real, demo, practice, or synthetic trading orders are permitted or implemented.
- Phase 33 authorizes authenticated instrument reads. It records wallet evidence at acquisition start. Wallet-specific operations are disabled. See [wallet contexts and protocol evidence](wallet_context_and_market_data.md).
- **Credential Hygiene:** Interactive foreground prompt only (`getpass`). Credentials are never saved to disk, `.env`, shell history, or version control.
- **No Unattended Execution:** Requires active operator presence throughout the session.
- **Fail-Closed Teardown:** Zero reconnect loops; any transport error or timeout immediately disconnects and shuts down.
- **Immutable Evidence:** Local storage only (`data/phase33/`); files are create-only and cannot be overwritten.
- **No Automatic Follow-up Runs:** The trial terminates upon completion or cancellation and does not schedule recurring jobs.

### Scope of Time Limits and Network Boundaries

The 180-second ceiling (`--duration 180`) strictly covers the **observation loop** (Step [3/4]). It does not claim to bound interactive typing time during credential entry. The exact phase bounds are:

1. **Interactive Credential Entry:** Operator-paced in the foreground; no network I/O occurs until credentials are submitted.
2. HTTPS login, WebSocket connection and authentication have separate timeouts. The configured connection and authentication timeouts are 15 seconds each.
3. Account discovery waits up to 10 seconds each for profile and balances. A missing or ambiguous selection is recorded in the evidence summary.
4. The 180-second observation budget includes pending candle requests and sleeps between observations. Operator interruption closes the connection. Logout and local evidence finalization run outside the observation budget.

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

When authorized, execute the measurement command in an interactive terminal session using **only** supported Typer CLI flags:

```powershell
.venv\Scripts\python.exe -m iqoption_research.cli.main timing measure `
  --live `
  --yes-i-understand `
  --max-candles 2 `
  --duration 180 `
  --output data/phase33/supervised_timing_evidence.json
```

### Supported CLI Flags

- `--live`: Enables read-only live broker measurement facade (interactive prompt; zero orders).
- `--yes-i-understand`: Explicit confirmation flag confirming operator is present and acknowledges operational bounds.
- `--max-candles 2`: Hard ceiling of two observed completed 1-minute candle boundaries (hard maximum: 5).
- `--duration 180`: Hard observation loop duration ceiling in seconds (hard maximum: 300).
- `--output data/phase33/supervised_timing_evidence.json`: Destination evidence path template. In live mode, the runner automatically creates a unique file embedding the session ID (`data/phase33/supervised_timing_evidence_meas_<session_id>.json`).
- `--overwrite`: **Strictly Forbidden in Live Mode.** Attempting to pass `--overwrite` with `--live` aborts immediately with an error.

> [!WARNING]
> Do **NOT** pass unsupported flags such as `--symbol`, `--size`, `--max-seconds`, `--poll-interval`, or `--out-dir`. The CLI will reject unrecognized options. EURUSD and 60-second candles are hardcoded defaults validated internally.

---

## 4. Step 3 — Provide Credentials Locally

1. When prompted by the CLI:
   ```text
   Enter IQ Option credentials (interactive prompt only):
   Broker Account Email: [hidden]
   Broker Account Password: [hidden]
   ```
2. Type the account email and press Enter.
3. Type or paste the account password. Password input is handled via `getpass` and characters will not echo on screen.
4. **Agent Boundary:** Never provide credentials to an AI assistant, chat interface, prompt, or repository file.

---

## 5. Step 4 — Supervise and Stop the Measurement

The measurement executes in the foreground, displaying stage checkpoints in real time:

```text
[1/4] Read-only candle acquisition. Wallet evidence: ACCOUNT_SELECTION_UNVERIFIED
[2/4] Initializing clock discipline anchor and stability checks...
      Anchor UTC epoch ns: 1795000000000000000
      NTP uncertainty budget: +/-25.00 ms

[3/4] Starting observation window loop (timer covers candle observation loop: max 180s)...

      Observing F=1795063200 (now=1795063202, offset=2s)...
      Candle F=1795063200 observed:
        - Client frame receipt:     2341.12 ms from candle close [compliant: YES (within 15s)]
        - Candle validation:        2356.45 ms from candle close [compliant: YES (within 15s)]
        - Full prediction contract: NOT_MEASURED (timing measurement only)
        - Observation verdict:      ACQUISITION_15S_COMPLIANT
```

### Session Termination and Cancellation Teardown

- **Normal Completion:** The session terminates automatically after acquiring 2 completed candle cycles or when the 180-second observation loop timer expires.
- **Graceful Manual Interruption (`Ctrl+C`):** The operator may abort at any time by pressing `Ctrl+C`. The session catches `KeyboardInterrupt` / `asyncio.CancelledError`, immediately disconnects from the WebSocket transport, executes clean shutdown, and saves whatever completed observations were acquired to the immutable evidence file.
- **Connection Interruption:** If the broker disconnects, drops the socket, or returns an error, the session fails closed immediately without reconnect loops.

---

## 6. Step 5 — Verify the Evidence File

Upon completion, the session prints the exact persisted destination, for example:
`data/phase33/supervised_timing_evidence_meas_<session_id>.json`

Verify the evidence file using the Phase 33 verification tool:

```powershell
.venv\Scripts\python.exe -m iqoption_research.cli.main timing verify-evidence `
  --path data/phase33/supervised_timing_evidence_meas_<session_id>.json
```

Or for machine-readable JSON:

```powershell
.venv\Scripts\python.exe -m iqoption_research.cli.main timing verify-evidence `
  --path data/phase33/supervised_timing_evidence_meas_<session_id>.json `
  --json
```

### Verification Checks Performed

- SHA-256 integrity hash verification over all observations and the registry envelope.
- Observation sequential index integrity (1-based, strictly monotonic, no skipped indices).
- Candle boundary uniqueness and strict chronological increase ($F_{i} > F_{i-1}$).
- Genuine request ID tracking: verifies request ID consistency across dispatch (`STAGE_1`), receipt (`STAGE_2`), parsing (`STAGE_3`), and validation (`STAGE_4`), and enforces uniqueness across observations.
- Strict provenance separation (rejection of synthetic data in live evidence; rejection of live claims in simulation).
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
