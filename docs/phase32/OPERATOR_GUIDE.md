# Phase 32: Operator-Supervised Acquisition Timing Measurement Guide

## 1. Safety Boundaries & Execution Rules

Before running any measurement commands, verify the following immutable rules:

1. **Foreground Supervision Required**: The operator must remain present in front of the console for the entire duration (maximum 300 seconds / 5 minutes).
2. **Interactive Credentials Only**: The tool **NEVER** reads credentials from `.env`, configuration files, or environment variables. Credentials are typed directly into the secure console prompt via `getpass`.
3. **Practice Account Only**: The system queries account status upon connection. If the account is not verified as `PRACTICE` (`active_balance.type == 4`), execution terminates immediately (`fail-closed`).
4. **Zero Broker Orders**: Outgoing trading frames are forbidden (`validate_outgoing_frame` enforced, `assert_read_only` guarded). Zero real or practice trades are placed.
5. **Fail-Closed Teardown**: If the network connection drops or an unexpected response arrives, the session aborts immediately without attempting reconnection loops.
6. **Graceful Cancellation**: Pressing `Ctrl+C` cleanly closes the WebSocket session, terminates the broker connection, and saves all collected timing observations up to that moment.

---

## 2. Pre-Measurement Verification (Offline)

Before attempting any network-facing operation, run the offline pipeline audit and replay test suite to confirm local system integrity:

```powershell
Set-Location -LiteralPath 'C:\Users\elmoc\Desktop\IqOptionBot'

# 1. Audit all collection pipeline stages and timestamp attribution
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main timing pipeline-audit

# 2. Run the deterministic offline replay simulation (10 scenarios)
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main timing replay

# 3. Execute bounded measurement in offline simulation mode
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main timing measure --simulation --output data/phase32/simulated_timing_evidence.json
```

---

## 3. Authorized Live Measurement Session (Foreground Only)

When explicitly authorized to perform empirical data-timing measurements on the live market feed:

```powershell
Set-Location -LiteralPath 'C:\Users\elmoc\Desktop\IqOptionBot'

# Live measurement command with explicit confirmation and safety bounds:
# - Maximum duration: 300 seconds (5 minutes)
# - Maximum completed candles: 5 cycles
# - Destination: data/phase32/supervised_timing_evidence.json
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main timing measure `
    --live `
    --yes-i-understand `
    --duration 300 `
    --max-candles 5 `
    --output data/phase32/supervised_timing_evidence.json
```

### Interactive Prompt Walkthrough

1. The console will display the safety banner and prompt for credentials:
   ```text
   Broker Account Email: your_practice_email@example.com
   Broker Account Password: [Hidden input]
   ```
2. The system verifies wallet authority:
   ```text
   [1/4] Verifying broker account and wallet authority...
         PRACTICE wallet verified successfully.
   ```
3. Clock discipline initializes:
   ```text
   [2/4] Initializing clock discipline anchor and stability checks...
         Anchor UTC epoch ns: 1795029000123456789
         NTP uncertainty budget: +/-25.00 ms
   ```
4. Observation window runs:
   ```text
   [3/4] Starting observation window (max 300s)...
         Waiting 12s for next boundary (F=1795029060)...
         Observing F=1795029060 (now=1795029062, offset=2s)...
         Candle F=1795029060 observed. Arrival offset: 1420.50 ms
         Acquisition 15s gate compliant: True (verdict: ACQUISITION_15S_COMPLIANT)
         Full Phase 30 prediction contract: NOT_MEASURED (timing measurement only)
   ```
5. Session terminates cleanly upon reaching 5 candles or 300 seconds.

---

## 4. Evidence Inspection & Verification

Once completed, inspect the generated evidence registry:

```powershell
# View formatted summary of the preserved evidence
Get-Content -LiteralPath 'data/phase32/supervised_timing_evidence.json' | ConvertFrom-Json | Select-Object -ExpandProperty summary

# Verify cryptographic integrity hash
Get-Content -LiteralPath 'data/phase32/supervised_timing_evidence.json' | ConvertFrom-Json | Select-Object session_id, operational_mode, integrity_hash
```

---

## 5. Emergency Interruption

If an unexpected condition occurs or you need to abort early:
- Press **Ctrl+C** once in the PowerShell terminal.
- The system will immediately catch the interrupt, clean up the WebSocket connection, close the session, and save partial records without corrupting state.
