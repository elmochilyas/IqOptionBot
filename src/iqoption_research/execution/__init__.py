"""Paper trading execution and safety control package (Phase 5)."""

from iqoption_research.execution.engine import (
    EngineStepResult,
    PaperTradingConfig,
    PaperTradingEngine,
)
from iqoption_research.execution.models import (
    VALID_ORDER_TRANSITIONS,
    ExecutionEvent,
    InvalidStateTransitionError,
    OrderSide,
    OrderState,
    PaperOrder,
    PaperPortfolio,
    ProductType,
)
from iqoption_research.execution.risk import (
    CircuitBreakerState,
    OperationalRiskManager,
    OrderProposal,
    RiskConfig,
    RiskDecision,
    RiskRejectionReason,
)
from iqoption_research.execution.safety import (
    BrokerExecutionForbiddenError,
    DisabledBrokerExecutionAdapter,
    PracticeAccountSafetyValidator,
    PracticeSafetyCheck,
    PracticeSafetyError,
    PracticeSafetyVerdict,
    RealAccountExecutionBlockedError,
    StaleVerificationBlockedError,
)
from iqoption_research.execution.state_machine import (
    DuplicateSignalError,
    ExecutionJournal,
    ExecutionStateMachine,
)

__all__ = [
    "VALID_ORDER_TRANSITIONS",
    "BrokerExecutionForbiddenError",
    "CircuitBreakerState",
    "DisabledBrokerExecutionAdapter",
    "DuplicateSignalError",
    "EngineStepResult",
    "ExecutionEvent",
    "ExecutionJournal",
    "ExecutionStateMachine",
    "InvalidStateTransitionError",
    "OperationalRiskManager",
    "OrderProposal",
    "OrderSide",
    "OrderState",
    "PaperOrder",
    "PaperPortfolio",
    "PaperTradingConfig",
    "PaperTradingEngine",
    "PracticeAccountSafetyValidator",
    "PracticeSafetyCheck",
    "PracticeSafetyError",
    "PracticeSafetyVerdict",
    "ProductType",
    "RealAccountExecutionBlockedError",
    "RiskConfig",
    "RiskDecision",
    "RiskRejectionReason",
    "StaleVerificationBlockedError",
]
