"""Result container for a candle fetch."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from iqoption_research.timing.taxonomy import BrokerClockDiagnostics

from iqoption_research.market_data.candles import CandleError
from iqoption_research.market_data.validation import DataQualityReport
from iqoption_research.models.market import Candle, CandleSize


def summarize_candle_response(
    response: dict[str, Any], request_id: str, active_id: int
) -> dict[str, object]:
    """Project transport evidence to fixed labels/numeric status codes, never payload text."""
    if not isinstance(response, dict):
        return {"frame_type": "INVALID", "correlation": "UNVERIFIED"}
    labels = {
        "ok",
        "success",
        "error",
        "failed",
        "fail",
        "denied",
        "unauthorized",
        "rate_limited",
        "no_data",
        "not_found",
    }

    def status(value: Any) -> str | int:
        if value is None:
            return "ABSENT"
        if type(value) is int and -9999 <= value <= 9999:
            return value
        if isinstance(value, str) and value.lower() in labels:
            return value.upper()
        return "PRESENT_UNRECOGNIZED"

    msg = response.get("msg")
    payload = msg if isinstance(msg, dict) else {}
    req = response.get("request_id")
    identity = payload.get("active_id")
    name = response.get("name")
    return {
        "frame_type": "CANDLES" if name == "candles" else "ERROR" if name == "error" else "OTHER",
        "correlation": "ABSENT"
        if req is None
        else "MATCHED"
        if str(req) == request_id
        else "MISMATCH",
        "envelope_identity": "ABSENT"
        if identity is None
        else "MATCHED"
        if str(identity) == str(active_id)
        else "MISMATCH",
        "envelope_status": status(response.get("status")),
        "payload_status": status(payload.get("status")),
        "envelope_code": status(response.get("code")),
        "payload_code": status(payload.get("code")),
        "error_field_present": any(
            key in container for container in (response, payload) for key in ("error", "errors")
        ),
        "error_signal_present": any(
            bool(container.get(key))
            for container in (response, payload)
            for key in ("error", "errors")
        ),
        "candles_list_present": isinstance(payload.get("candles"), list),
    }


@dataclass(frozen=True, slots=True)
class CandleFetchResult:
    """Validated candles plus data-quality information.

    ``source`` distinguishes real broker data ("iqoption") from offline
    simulation ("simulated"). Simulated data must never be presented as
    IQ Option market data.
    """

    symbol: str
    size: CandleSize
    candles: tuple[Candle, ...]
    errors: tuple[CandleError, ...]
    quality: DataQualityReport
    fetched_at: datetime
    raw_count: int
    source: str
    completion_clock_ns: int | None = None
    response_evidence: dict[str, object] | None = None
    recv_mono_ns: int | None = None
    parsed_mono_ns: int | None = None
    request_completion_mono_ns: int | None = None
    request_id: str | None = None
    server_time_fresh: bool | None = None
    clock_diagnostics: BrokerClockDiagnostics | None = None

    @property
    def candle_list(self) -> list[Candle]:
        return list(self.candles)

    def to_dict(self, *, include_candles: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "symbol": self.symbol,
            "size_seconds": self.size.value,
            "source": self.source,
            "fetched_at": self.fetched_at.isoformat(),
            "raw_count": self.raw_count,
            "valid_count": len(self.candles),
            "error_count": len(self.errors),
            "errors": [
                {"index": e.index, "reason": e.reason, "detail": e.detail} for e in self.errors
            ],
            "quality": self.quality.to_dict(),
        }
        if include_candles:
            payload["candles"] = [
                {
                    "from_utc": c.timestamp_utc.isoformat(),
                    "to_utc": c.timestamp_end_utc.isoformat(),
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                    "bid": c.bid,
                    "ask": c.ask,
                    "is_complete": c.is_complete,
                }
                for c in self.candles
            ]
        return payload
