"""Safe protocol diagnostic recorder for IQ Option WebSocket communication.

Collects structural frame metadata, message counts, and lifecycle flags without
storing or exposing any sensitive values, tokens, credentials, or personal data.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from iqoption_research.models.connection import ConnectionState


@dataclass(frozen=True, slots=True)
class FrameShape:
    """Sanitized structural metadata for an incoming frame.

    Contains NO sensitive values, tokens, cookies, profile details, or amounts.
    """

    name: str
    payload_type: str
    keys: tuple[str, ...]
    length: int | None
    received_at: datetime

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "type": self.payload_type,
            "received_at": self.received_at.isoformat(),
        }
        if self.keys:
            result["keys"] = list(self.keys)
        if self.length is not None:
            result["length"] = self.length
        return result


class ProtocolDiagnostics:
    """Safe protocol diagnostic recorder.

    Disabled by default. Provides metadata-only visibility into WebSocket
    message flow to diagnose missing frames vs. parsing failures.
    """

    def __init__(self, *, enabled: bool = False, max_history: int = 100) -> None:
        self.enabled = enabled
        self._counts: Counter[str] = Counter()
        self._history: deque[FrameShape] = deque(maxlen=max_history)
        self._unhandled_names: set[str] = set()
        self._parsing_failures: list[dict[str, str]] = []
        self._auth_received = False
        self._auth_ok = False
        self._profile_received = False
        self._balances_received = False
        self._failure_category: str | None = None
        self._failure_detail: str | None = None

    def record_incoming(self, name: str | None, msg: Any, *, handled: bool = True) -> None:
        if not self.enabled:
            return
        safe_name = str(name) if name is not None else "UNKNOWN"
        self._counts[safe_name] += 1
        if not handled:
            self._unhandled_names.add(safe_name)

        keys: tuple[str, ...] = ()
        length: int | None = None

        if isinstance(msg, dict):
            # Only record key names, never values
            keys = tuple(sorted(str(k) for k in msg))
        elif isinstance(msg, list):
            length = len(msg)
            sample_keys: set[str] = set()
            for item in msg[:5]:
                if isinstance(item, dict):
                    sample_keys.update(str(k) for k in item)
            if sample_keys:
                keys = tuple(sorted(sample_keys))

        shape = FrameShape(
            name=safe_name,
            payload_type=type(msg).__name__,
            keys=keys,
            length=length,
            received_at=datetime.now(UTC),
        )
        self._history.append(shape)

    def record_auth_status(self, is_ok: bool) -> None:
        if not self.enabled:
            return
        self._auth_received = True
        self._auth_ok = is_ok

    def record_profile_status(self, received: bool) -> None:
        if not self.enabled:
            return
        self._profile_received = received

    def record_balances_status(self, received: bool) -> None:
        if not self.enabled:
            return
        self._balances_received = received

    def record_parsing_failure(self, frame_name: str, exception_type: str) -> None:
        if not self.enabled:
            return
        self._parsing_failures.append({
            "frame": frame_name,
            "error_type": exception_type,
        })

    def record_failure(self, category: str, detail: str = "") -> None:
        if not self.enabled:
            return
        self._failure_category = category
        self._failure_detail = detail

    def summary(self, connection_state: ConnectionState | None = None) -> dict[str, Any]:
        return {
            "diagnostics_enabled": self.enabled,
            "total_messages": sum(self._counts.values()),
            "message_counts": dict(self._counts),
            "unhandled_message_names": sorted(self._unhandled_names),
            "parsing_failures": list(self._parsing_failures),
            "authentication_state": {
                "received": self._auth_received,
                "authenticated": self._auth_ok,
            },
            "profile_received": self._profile_received,
            "balances_received": self._balances_received,
            "failure_category": self._failure_category,
            "connection_state": connection_state.value if connection_state else None,
            "recent_frame_shapes": [shape.to_dict() for shape in self._history],
        }
