"""Modular product execution adapters package (Phase 16)."""

from iqoption_research.execution.adapters.base import (
    BaseExecutionAdapter,
    SettlementOutcome,
    SizingResult,
)
from iqoption_research.execution.adapters.binary import (
    BinaryOptionsAdapter,
    build_binary_order_frame,
)
from iqoption_research.execution.adapters.digital import (
    DigitalOptionsAdapter,
    DigitalPayoffModel,
)
from iqoption_research.execution.adapters.forex import (
    ForexCFDAdapter,
    ForexPayoffModel,
)

__all__ = [
    "BaseExecutionAdapter",
    "BinaryOptionsAdapter",
    "DigitalOptionsAdapter",
    "DigitalPayoffModel",
    "ForexCFDAdapter",
    "ForexPayoffModel",
    "SettlementOutcome",
    "SizingResult",
    "build_binary_order_frame",
]
