"""Offline audit: only derived outputs in this directory; never resolves outcomes."""

from pathlib import Path
import collections, csv, hashlib, json, math, socket, sys
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def deny_network(*args, **kwargs):
    raise RuntimeError("Network prohibited during audit")


class OfflineSocket(socket.socket):
    def connect(self, *args, **kwargs):
        return deny_network()

    def connect_ex(self, *args, **kwargs):
        return deny_network()

    def sendto(self, *args, **kwargs):
        return deny_network()


socket.socket = OfflineSocket
socket.create_connection = deny_network
from iqoption_research.shadow.observer import verify_session_integrity
from iqoption_research.ml.laboratory import ModelLaboratory
from iqoption_research.ml.decision.policy import classify_directional_decision

SIDS = ["pobs_d2d1a79b15d8", "pobs_7843aa5dbb1c"]
regpath = ROOT / "data/shadow_eval/registry.json"
registry = json.loads(regpath.read_text())
models = list(registry["records"])
protected = [regpath]
for sid in SIDS:
    protected += list((ROOT / "data/shadow_eval/sessions" / sid).iterdir())
for mid in models:
    protected += list((ROOT / "data/candidate_models" / mid).iterdir())


def hashes():
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in protected
        if p.is_file()
    }


before = hashes()


def wilson(w, n):
    if not n:
        return None
    z = 1.959963984540054
    p = w / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [center - half, center + half]


def dist(x):
    return dict(
        zip(
            ["min", "q25", "median", "q75", "max", "mean"],
            [*map(float, np.quantile(x, [0, 0.25, 0.5, 0.75, 1])), float(np.mean(x))],
        )
    )


def iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


result = {"verification": {}, "artifacts": {}, "sessions": {}, "checks": {}}
# Avoid ModelLaboratory.__init__: it initializes the experiment tracker/database.
# This read-only loader only requires root, and verifies manifest/bundle hashes before load.
lab = object.__new__(ModelLaboratory)
lab.root = ROOT
for mid in models:
    b = lab.load_candidate_bundle(mid)
    m = json.loads((ROOT / "data/candidate_models" / mid / "manifest.json").read_text())
    rec = registry["records"][mid]
    assert m["bundle_sha256"] == rec["bundle_sha256"]
    assert m["manifest_sha256"] == rec["manifest_sha256"]
    a = {
        "bundle_verified": True,
        "registered_at": rec["registered_at_utc"],
        "bundle_threshold": b["config"]["confidence_threshold"],
        "registered_threshold": rec["confidence_threshold"],
        "bundle_sha256": rec["bundle_sha256"],
        "data_cutoff_ts": rec["data_cutoff_ts"],
    }
    if "majority_class_" in vars(b["model"]):
        raw = b["model"].predict_proba(np.zeros((1, len(b["feature_names"]))))
        calibrated = b["calibrator"].calibrate(raw)
        a.update(
            raw_prob_up=float(raw[0, 1]),
            calibrated_prob_up=float(calibrated[0, 1]),
            frozen_direction="UP" if b["model"].majority_class_ == 1 else "DOWN",
        )
    result["artifacts"][mid] = a

rows = []
failures = []


def check(ok, label, pid):
    if not ok:
        failures.append([label, pid])


for sid in SIDS:
    folder = ROOT / "data/shadow_eval/sessions" / sid
    v = verify_session_integrity(folder)
    result["verification"][sid] = v
    assert (
        v["is_valid"]
        and v["is_prospective_verified"]
        and v["evidence_classification"] == "PROSPECTIVE_SHADOW"
    )

    def read(name):
        return [json.loads(x) for x in (folder / name).read_text().splitlines() if x]

    ps, cs, os = read("predictions.jsonl"), read("commitments.jsonl"), read("outcomes.jsonl")
    cm = {c["prediction_id"]: c for c in cs}
    om = {o["prediction_id"]: o for o in os}
    assert len(cm) == len(cs) == len(ps) and len(om) == len(os)
    meta = json.loads((folder / "session.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    result["sessions"][sid] = {
        "start": meta["created_at_utc"],
        "end": meta["ended_at_utc"],
        "duration": meta["actual_duration_seconds"],
        "operational": {
            k: v
            for k, v in manifest.items()
            if k.endswith("_count")
            or k
            in [
                "warmup_state",
                "consecutive_valid_candles",
                "total_candles_processed",
                "termination_reason",
            ]
        },
    }
    assert (
        meta["execution_authority"] == "NONE"
        and meta["disposition"] == "NO_DEMONSTRATED_EDGE"
        and meta["assumed_payout"] == 0.85
    )
    for line, p in enumerate(ps, 1):
        pid = p["prediction_id"]
        c = cm[pid]
        o = om.get(pid)
        t = p["feature_cutoff_ts"]
        maturity = p["expected_maturity_ts"]
        check(
            p["evidence_class"] == "PROSPECTIVE_SHADOW"
            and not p["runtime_diagnostics"]["is_synthetic"]
            and p["provenance_class"] == "supervised_live_collector",
            "genuine",
            pid,
        )
        check(
            p["artifact_hash"] == result["artifacts"][p["model_id"]]["bundle_sha256"],
            "artifact",
            pid,
        )
        check(
            p["confidence_threshold"] == registry["records"][p["model_id"]]["confidence_threshold"],
            "threshold",
            pid,
        )
        check(
            classify_directional_decision(p["probabilities"]["UP"], p["confidence_threshold"]).value
            == p["decision"],
            "decision",
            pid,
        )
        check(abs(sum(p["probabilities"].values()) - 1) < 1e-6, "probability_sum", pid)
        check(maturity == t + 60, "horizon", pid)
        check(
            abs(c["commitment_deadline_ts"] - (maturity - 5 - c["clock_uncertainty_ms"] / 1000))
            < 1e-5,
            "deadline_formula",
            pid,
        )
        receipt = datetime.fromisoformat(p["data_availability_utc"]).timestamp()
        created = datetime.fromisoformat(p["prediction_created_ts"]).timestamp()
        check(
            t <= receipt <= created <= c["post_fsync_completion_ts"] < c["commitment_deadline_ts"],
            "time_order",
            pid,
        )
        if o:
            direction = (
                "UP"
                if o["maturity_candle_close"] > o["cutoff_candle_close"]
                else "DOWN"
                if o["maturity_candle_close"] < o["cutoff_candle_close"]
                else "TIE"
            )
            status = (
                "SKIPPED_ABSTAIN"
                if p["decision"] == "SKIP"
                else "RESOLVED_TIE"
                if direction == "TIE"
                else "RESOLVED_WIN"
                if direction == p["decision"]
                else "RESOLVED_LOSS"
            )
            check(
                direction == o["market_direction"] and status == o["outcome_status"],
                "outcome_direction_status",
                pid,
            )
            check(
                round(
                    (o["maturity_candle_close"] - o["cutoff_candle_close"])
                    / o["cutoff_candle_close"],
                    8,
                )
                == o["actual_return"],
                "return",
                pid,
            )
            check(
                datetime.fromisoformat(o["evaluated_at_utc"]).timestamp() >= maturity,
                "resolution_time",
                pid,
            )
            pnl = (
                None
                if status == "SKIPPED_ABSTAIN"
                else 0.85
                if status == "RESOLVED_WIN"
                else 0
                if status == "RESOLVED_TIE"
                else -1
            )
            check(
                o["simulated_payout_pnl"] == pnl
                and o["was_trade_taken"] == (p["decision"] != "SKIP"),
                "pnl_trade_flag",
                pid,
            )
        rows.append({
            **p,
            **(o or {}),
            "prediction_line": line,
            "commitment": c,
            "eligible": bool(
                o
                and c["is_timely"]
                and o["outcome_status"]
                in ["RESOLVED_WIN", "RESOLVED_LOSS", "RESOLVED_TIE", "SKIPPED_ABSTAIN"]
            ),
            "deadline_slack_s": c["commitment_deadline_ts"] - c["post_fsync_completion_ts"],
            "receipt_to_fsync_ms": (c["post_fsync_completion_ts"] - receipt) * 1000,
        })


def stats(rs):
    eligible = [r for r in rs if r["eligible"]]
    trades = [r for r in eligible if r["decision"] != "SKIP"]
    decisive = [r for r in trades if r["market_direction"] != "TIE"]
    labels = [r for r in eligible if r["market_direction"] != "TIE"]
    w = sum(r["outcome_status"] == "RESOLVED_WIN" for r in trades)
    l = sum(r["outcome_status"] == "RESOLVED_LOSS" for r in trades)
    ties = len(trades) - w - l
    conf = [max(r["probabilities"].values()) for r in rs]
    s = {
        "total": len(rs),
        "decisions": dict(collections.Counter(r["decision"] for r in rs)),
        "statuses": dict(collections.Counter(r["outcome_status"] for r in rs)),
        "timely": sum(r["commitment"]["is_timely"] for r in rs),
        "eligible": len(eligible),
        "wins": w,
        "losses": l,
        "ties": ties,
        "trades": len(trades),
        "coverage": len(trades) / len(eligible),
        "coverage_ci": wilson(len(trades), len(eligible)),
        "accuracy": w / len(decisive) if decisive else None,
        "accuracy_ci": wilson(w, len(decisive)),
        "binary_expectancy": (0.85 * w - l) / len(trades) if trades else None,
        "digital_expectancy": (0.85 * w - l - ties) / len(trades) if trades else None,
        "total_binary_units": 0.85 * w - l,
        "confidence": dist(conf),
        "p_up": dist([r["probabilities"]["UP"] for r in rs]),
        "label_counts": dict(collections.Counter(r["market_direction"] for r in eligible)),
        "calibration_n": len(labels),
        "mean_prob_up": float(np.mean([r["probabilities"]["UP"] for r in labels])),
        "up_rate": sum(r["market_direction"] == "UP" for r in labels) / len(labels),
        "brier": float(
            np.mean([
                (r["probabilities"]["UP"] - (r["market_direction"] == "UP")) ** 2 for r in labels
            ])
        ),
        "log_loss": float(
            np.mean([-math.log(r["probabilities"][r["market_direction"]]) for r in labels])
        ),
        "argmax_correct": sum(
            ("UP" if r["probabilities"]["UP"] >= 0.5 else "DOWN") == r["market_direction"]
            for r in labels
        ),
        "confusion": dict(
            collections.Counter(r["decision"] + " / " + r["market_direction"] for r in eligible)
        ),
        "feature_counts": dict(
            collections.Counter(r["runtime_diagnostics"]["feature_count"] for r in rs)
        ),
        "unique_feature_fingerprints": len(set(r["feature_fingerprint"] for r in rs)),
        "latency_ms": dist([r["runtime_diagnostics"]["latency_ms"] for r in rs]),
        "inference_ms": dist([r["runtime_diagnostics"]["inference_duration_ms"] for r in rs]),
        "uncertainty_ms": dist([r["commitment"]["clock_uncertainty_ms"] for r in rs]),
        "deadline_slack_s": dist([r["deadline_slack_s"] for r in rs]),
        "receipt_to_fsync_ms": dist([r["receipt_to_fsync_ms"] for r in rs]),
        "cutoff_range": [
            iso(min(r["feature_cutoff_ts"] for r in rs)),
            iso(max(r["feature_cutoff_ts"] for r in rs)),
        ],
        "confidence_bins": [],
    }
    for lo, hi in [(0.5, 0.52), (0.52, 0.54), (0.54, 0.56), (0.56, 1.000001)]:
        rr = [r for r in labels if lo <= max(r["probabilities"].values()) < hi]
        if rr:
            correct = sum(
                ("UP" if r["probabilities"]["UP"] >= 0.5 else "DOWN") == r["market_direction"]
                for r in rr
            )
            s["confidence_bins"].append({
                "lower": lo,
                "upper": hi,
                "n": len(rr),
                "mean_confidence": float(np.mean([max(r["probabilities"].values()) for r in rr])),
                "correct": correct,
                "ci": wilson(correct, len(rr)),
            })
    return s


result["statistics"] = {}
result["matched"] = {}
for sid in SIDS + ["pooled"]:
    rr = [r for r in rows if sid == "pooled" or r["session_id"] == sid]
    result["statistics"][sid] = {
        mid: stats([r for r in rr if r["model_id"] == mid]) for mid in models
    }
    lr = [r for r in rr if r["model_id"] == models[0] and r["eligible"]]
    bm = {
        (r["session_id"], r["feature_cutoff_ts"]): r
        for r in rr
        if r["model_id"] == models[1] and r["eligible"]
    }
    pairs = [(r, bm[(r["session_id"], r["feature_cutoff_ts"])]) for r in lr]
    for a, b in pairs:
        check(
            all(
                a[k] == b[k]
                for k in [
                    "market_direction",
                    "cutoff_candle_close",
                    "maturity_candle_close",
                    "feature_fingerprint",
                ]
            ),
            "matched_evidence",
            a["prediction_id"],
        )
    result["matched"][sid] = {}
    for subset in ["all_decisive", "candidate_selected"]:
        pp = [
            (a, b)
            for a, b in pairs
            if a["market_direction"] != "TIE"
            and (subset == "all_decisive" or a["decision"] != "SKIP")
        ]
        cells = collections.Counter()
        for a, b in pp:
            ca = ("UP" if a["probabilities"]["UP"] >= 0.5 else "DOWN") == a["market_direction"]
            ba = "DOWN" == a["market_direction"]
            cells[str(ca) + " / " + str(ba)] += 1
        n = len(pp)
        bw = sum(b["market_direction"] == "DOWN" for a, b in pp)
        cw = sum(
            ("UP" if a["probabilities"]["UP"] >= 0.5 else "DOWN") == a["market_direction"]
            for a, b in pp
        )
        result["matched"][sid][subset] = {
            "n": n,
            "candidate_correct": cw,
            "baseline_correct": bw,
            "baseline_ci": wilson(bw, n),
            "paired_cells": dict(cells),
            "baseline_brier": float(
                np.mean([
                    (b["probabilities"]["UP"] - (b["market_direction"] == "UP")) ** 2 for a, b in pp
                ])
            ),
            "candidate_brier": float(
                np.mean([
                    (a["probabilities"]["UP"] - (a["market_direction"] == "UP")) ** 2 for a, b in pp
                ])
            ),
        }
keys = [
    (r["session_id"], r["model_id"], r["symbol"], r["timeframe"], r["feature_cutoff_ts"])
    for r in rows
]
result["checks"]["duplicate_keys"] = len(keys) - len(set(keys))
sets = [set(r["feature_cutoff_ts"] for r in rows if r["session_id"] == sid) for sid in SIDS]
result["checks"]["overlap_cutoffs"] = sorted(sets[0] & sets[1])
for sid in SIDS:
    rr = [r for r in rows if r["session_id"] == sid and r["model_id"] == models[0]]
    for a, b in zip(rr, rr[1:]):
        check(
            b["feature_cutoff_ts"] - a["feature_cutoff_ts"] == 60,
            "contiguous_cutoffs",
            a["prediction_id"],
        )
        if a["eligible"] and b["eligible"]:
            check(
                a["maturity_candle_close"] == b["cutoff_candle_close"],
                "adjacent_prices",
                a["prediction_id"],
            )
    result["sessions"][sid]["logistic_probability_counts"] = dict(
        collections.Counter(r["probabilities"]["UP"] for r in rr)
    )
    result["sessions"][sid]["pending"] = [
        {
            k: r[k]
            for k in ["prediction_id", "feature_cutoff_utc", "expected_maturity_utc", "decision"]
        }
        for r in rr
        if not r["eligible"]
    ]
snapshot = ROOT / "data/research_datasets/rds_eurusd_m1_20261121_161531_215496a1a71f/manifest.json"
sm = json.loads(snapshot.read_text())
result["snapshot_metadata"] = {
    k: sm[k] for k in ["from_utc", "to_utc", "created_at_utc", "sha256_checksum"]
}
result["checks"]["failures"] = failures
result["checks"]["protected_files_unchanged"] = before == hashes()
result["protected_hashes"] = before
assert not failures and before == hashes()
(OUT / "audit_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
fields = [
    "session_id",
    "prediction_id",
    "model_id",
    "prediction_line",
    "feature_cutoff_utc",
    "expected_maturity_utc",
    "decision",
    "outcome_status",
    "market_direction",
    "eligible",
    "deadline_slack_s",
    "receipt_to_fsync_ms",
    "feature_fingerprint",
]
with (OUT / "joined_observations.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields + ["p_up", "timely"])
    writer.writeheader()
    for r in rows:
        writer.writerow({
            **{k: r.get(k) for k in fields},
            "p_up": r["probabilities"]["UP"],
            "timely": r["commitment"]["is_timely"],
        })
print(
    json.dumps(
        {k: result[k] for k in ["artifacts", "sessions", "statistics", "matched", "checks"]},
        indent=2,
    )
)
