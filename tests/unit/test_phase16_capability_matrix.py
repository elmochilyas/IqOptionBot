"""Unit tests for Phase 16 Broker Capability Matrix and Operation Classification."""

from __future__ import annotations

from iqoption_research.execution.capability_matrix import (
    BROKER_CAPABILITIES,
    BrokerCapabilityMatrix,
    BrokerCapabilityStatus,
    OperationCategory,
)
from iqoption_research.execution.models import ProductType


def test_capability_matrix_initialization_and_counts() -> None:
    matrix = BrokerCapabilityMatrix()
    summary = matrix.summary_counts()

    assert summary[BrokerCapabilityStatus.IMPLEMENTED_AND_VERIFIED.value] >= 8
    assert summary[BrokerCapabilityStatus.IMPLEMENTED_SIMULATED_ONLY.value] >= 3
    assert summary[BrokerCapabilityStatus.UNVERIFIED_PROTOCOL_EVIDENCE.value] >= 3
    assert summary[BrokerCapabilityStatus.UNSUPPORTED.value] >= 6


def test_capability_matrix_filtering() -> None:
    matrix = BrokerCapabilityMatrix()

    # Filter by product
    binary_ops = matrix.filter_operations(product_type=ProductType.BINARY)
    assert len(binary_ops) >= 4
    for op in binary_ops:
        assert op.product_type == ProductType.BINARY

    digital_ops = matrix.filter_operations(product_type=ProductType.DIGITAL)
    assert len(digital_ops) >= 3
    for op in digital_ops:
        assert op.product_type == ProductType.DIGITAL

    forex_ops = matrix.filter_operations(product_type=ProductType.FOREX)
    assert len(forex_ops) >= 3
    for op in forex_ops:
        assert op.product_type == ProductType.FOREX

    # Filter by category
    auth_ops = matrix.filter_operations(category=OperationCategory.AUTHENTICATION)
    assert len(auth_ops) >= 3
    for op in auth_ops:
        assert op.category == OperationCategory.AUTHENTICATION


def test_live_broker_permissions_strictness() -> None:
    matrix = BrokerCapabilityMatrix()

    # Market data & Auth are verified and permitted live
    assert matrix.is_operation_permitted("auth_https_login", live_broker=True) is True
    assert matrix.is_operation_permitted("auth_ws_authenticate", live_broker=True) is True
    assert matrix.is_operation_permitted("market_get_candles", live_broker=True) is True
    assert matrix.is_operation_permitted("market_subscribe_candles", live_broker=True) is True

    # Real order placement is strictly prohibited
    assert matrix.is_operation_permitted("order_submission_real", live_broker=True) is False
    assert matrix.is_operation_permitted("order_submission_real", live_broker=False) is False

    # Automatic balance switching is strictly prohibited
    assert matrix.is_operation_permitted("balance_switch_auto", live_broker=True) is False

    # Practice binary order submission is unverified in live broker
    assert (
        matrix.is_operation_permitted("binary_order_submission_practice", live_broker=True) is False
    )

    # Simulated trades are permitted offline, but not live
    assert matrix.is_operation_permitted("binary_simulated_trade", live_broker=False) is True
    assert matrix.is_operation_permitted("binary_simulated_trade", live_broker=True) is False


def test_capability_matrix_markdown_and_dict_serialization() -> None:
    matrix = BrokerCapabilityMatrix()
    md_table = matrix.to_markdown_table()
    assert "| Operation ID | Product | Category |" in md_table
    assert "`auth_https_login`" in md_table
    assert "`order_submission_real`" in md_table

    data_list = matrix.to_dict()
    assert isinstance(data_list, list)
    assert len(data_list) == len(BROKER_CAPABILITIES)
    op0 = data_list[0]
    assert "operation_id" in op0
    assert "status" in op0
    assert "live_broker_allowed" in op0
