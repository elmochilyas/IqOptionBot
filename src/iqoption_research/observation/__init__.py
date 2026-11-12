"""Market observation subsystem for Phase 6.

Provides clock drift verification, feed heartbeat cadence monitoring,
reconnection resynchronization, and market observation adapters.
"""

from iqoption_research.observation.adapter import (
    MarketObservationAdapter,
    ObservationAuthorizationToken,
    ObservationEvent,
    ObservationMode,
    ObservationSecurityError,
)
from iqoption_research.observation.clock import (
    ClockDriftResult,
    ClockDriftValidator,
    ClockDriftVerdict,
)
from iqoption_research.observation.heartbeat import (
    FeedHealthEvent,
    FeedHealthStatus,
    FeedHeartbeatMonitor,
)
from iqoption_research.observation.reconnect import (
    ReconnectEvent,
    ReconnectResynchronizer,
    ResyncStatus,
)

__all__ = [
    "ClockDriftResult",
    "ClockDriftValidator",
    "ClockDriftVerdict",
    "FeedHealthEvent",
    "FeedHealthStatus",
    "FeedHeartbeatMonitor",
    "MarketObservationAdapter",
    "ObservationAuthorizationToken",
    "ObservationEvent",
    "ObservationMode",
    "ObservationSecurityError",
    "ReconnectEvent",
    "ReconnectResynchronizer",
    "ResyncStatus",
]
