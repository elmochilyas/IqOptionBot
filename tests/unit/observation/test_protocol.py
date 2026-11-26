"""Unit tests for Phase 17 Binary Practice protocol schemas and instrument schedules.

Validates:
- Strict Pydantic schema validation (extra='forbid') for outgoing binary order frames.
- Rejection of invalid types, negative amounts, extra attributes, and malformed frames.
- Parsing of broker order acknowledgements and settlement payloads.
- Binary instrument schedule constraints (minimum $1.00 stake, expiration durations).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from iqoption_research.execution.practice_transport import (
    MIN_PRACTICE_STAKE,
    BinaryInstrumentSchedule,
    IncomingBinaryOrderAckBody,
    IncomingBinarySettlementBody,
    OutgoingBinaryOrderRequestBody,
    OutgoingBinaryOrderRequestFrame,
    OutgoingBinaryOrderRequestMsg,
)


def test_outgoing_binary_order_request_valid() -> None:
    """Test valid outgoing binary option order request frame construction."""
    body = OutgoingBinaryOrderRequestBody(
        user_balance_id=123456,
        active_id=1,
        direction="call",
        price=1.0,
        exp=1700000060,
        type_id=1,
    )
    msg = OutgoingBinaryOrderRequestMsg(body=body)
    frame = OutgoingBinaryOrderRequestFrame(msg=msg, request_id="ord_req_001")

    dumped = frame.model_dump()
    assert dumped["name"] == "sendMessage"
    assert dumped["msg"]["name"] == "binary-options.open-option"
    assert dumped["msg"]["body"]["user_balance_id"] == 123456
    assert dumped["msg"]["body"]["active_id"] == 1
    assert dumped["msg"]["body"]["direction"] == "call"
    assert dumped["msg"]["body"]["price"] == 1.0
    assert dumped["msg"]["body"]["exp"] == 1700000060
    assert dumped["request_id"] == "ord_req_001"


def test_outgoing_binary_order_request_rejects_extra_fields() -> None:
    """Test that unexpected extra fields are forbidden by schema."""
    with pytest.raises(ValidationError):
        OutgoingBinaryOrderRequestBody(
            user_balance_id=123456,
            active_id=1,
            direction="call",
            price=1.0,
            exp=1700000060,
            type_id=1,
            unauthorized_field="malicious",  # type: ignore[call-arg]
        )


def test_outgoing_binary_order_request_rejects_sub_minimum_stake() -> None:
    """Test that stakes below MIN_PRACTICE_STAKE ($1.00) are rejected."""
    with pytest.raises(ValidationError):
        OutgoingBinaryOrderRequestBody(
            user_balance_id=123456,
            active_id=1,
            direction="call",
            price=0.50,  # Below $1.00
            exp=1700000060,
            type_id=1,
        )


def test_outgoing_binary_order_request_rejects_invalid_direction() -> None:
    """Test that invalid direction string is rejected."""
    with pytest.raises(ValidationError):
        OutgoingBinaryOrderRequestBody(
            user_balance_id=123456,
            active_id=1,
            direction="buy",  # Invalid; must be "call" or "put"  # type: ignore[arg-type]
            price=1.0,
            exp=1700000060,
            type_id=1,
        )


def test_incoming_binary_order_ack_parsing() -> None:
    """Test parsing of incoming option-opened acknowledgement."""
    payload = {
        "id": 987654321,
        "active_id": 1,
        "user_balance_id": 123456,
        "price": 1.0,
        "direction": "call",
        "exp": 1700000060,
        "created": 1700000000,
    }
    ack = IncomingBinaryOrderAckBody.model_validate(payload)
    assert ack.id == 987654321
    assert ack.active_id == 1
    assert ack.price == 1.0


def test_incoming_binary_settlement_parsing() -> None:
    """Test parsing of incoming option-closed settlement payload."""
    payload = {
        "option_id": "opt_9999",
        "status": "win",
        "profit": 0.85,
        "close_price": 1.0858,
        "close_time": 1700000060,
    }
    settle = IncomingBinarySettlementBody.model_validate(payload)
    assert settle.option_id == "opt_9999"
    assert settle.status == "win"
    assert settle.profit == 0.85
    assert settle.close_price == 1.0858


def test_binary_instrument_schedule_properties() -> None:
    """Test BinaryInstrumentSchedule limits and validations."""
    sched = BinaryInstrumentSchedule(
        symbol="EURUSD",
        active_id=1,
        min_stake=MIN_PRACTICE_STAKE,
        max_stake=100.0,
        payout_rate=0.85,
    )
    assert sched.symbol == "EURUSD"
    assert sched.active_id == 1
    assert sched.min_stake == 1.0
    assert sched.is_available is True
    assert 60 in sched.valid_durations_seconds
