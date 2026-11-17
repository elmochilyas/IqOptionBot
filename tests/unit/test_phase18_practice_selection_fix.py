"""Comprehensive unit and integration tests for Phase 18 Practice Selection Fix.

Covers:
1. Deep active balance extraction from nested structures (msg.result, msg.profile, etc.).
2. Non-destructive partial profile updates (partial profile frame without balance_id preserves it).
3. Dynamic handling of balance-changed push notification frames (updates active balance ID).
4. Delayed balance propagation in MockBrokerPracticeTransport (bounded verification loop).
5. Delayed balance propagation in Live Practice Transport with MockWebSocket (0 duplicate frames).
6. Verification timeout handling (fails closed when balance never switches to practice).
7. Disconnect / session loss during verification (fails closed with ConnectionError).
8. CLI practice-test --preflight-only mode (executes preflight checks, bypasses submission).
9. CLI practice-test failure handling (fails closed, displays clean Rich panel).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.execution.live_practice_transport import (
    IQOptionLivePracticeTransport,
    IQOptionLivePracticeTransportConfig,
)
from iqoption_research.execution.modes import PracticeAuthorizationError
from iqoption_research.execution.practice_transport import MockBrokerPracticeTransport
from iqoption_research.models.account import AccountType, Balance, Profile
from tests.unit.test_phase18_iqoption_live_transport import MockWebSocket

runner = CliRunner()


# ===========================================================================
# 1. Deep Balance Extraction Tests
# ===========================================================================
def test_deep_balance_extraction_from_nested_structures() -> None:
    """Validate extraction across all observed IQ Option protocol variants."""
    transport = IQOptionLivePracticeTransport(ssid="test_ssid")

    # Variant A: Wrapped under msg.result.balance_id
    frame_a = {
        "name": "profile",
        "msg": {
            "result": {
                "user_id": 12345,
                "balance_id": 101,
                "balance_type": 4,
            }
        },
    }
    extracted_a = transport._extract_active_balance_info(frame_a, source_frame="profile")
    assert extracted_a.balance_id == "101"
    assert extracted_a.account_type == AccountType.PRACTICE
    assert extracted_a.is_practice is True
    assert extracted_a.location == "msg.result.balance_id"

    # Variant B: Wrapped under msg.profile.balanceId
    frame_b = {
        "name": "profile",
        "msg": {
            "profile": {
                "userId": 12345,
                "balanceId": 202,
                "type": 1,
            }
        },
    }
    extracted_b = transport._extract_active_balance_info(frame_b, source_frame="profile")
    assert extracted_b.balance_id == "202"
    assert extracted_b.account_type == AccountType.REAL
    assert extracted_b.is_practice is False
    assert extracted_b.location == "msg.profile.balanceId"

    # Variant C: Wrapped under msg.current_balance
    frame_c = {
        "name": "profile",
        "msg": {
            "current_balance": {
                "id": 303,
                "type": 4,
            }
        },
    }
    extracted_c = transport._extract_active_balance_info(frame_c, source_frame="profile")
    assert extracted_c.balance_id == "303"
    assert extracted_c.account_type == AccountType.PRACTICE
    assert extracted_c.is_practice is True
    assert extracted_c.location == "msg.current_balance.id"

    # Variant D: Top-level active_balance_id
    frame_d = {
        "name": "profile",
        "msg": {
            "active_balance_id": 404,
            "balance_type": 4,
        },
    }
    extracted_d = transport._extract_active_balance_info(frame_d, source_frame="profile")
    assert extracted_d.balance_id == "404"
    assert extracted_d.is_practice is True
    assert extracted_d.location == "msg.active_balance_id"

    # Variant E: Balances array with is_active flag
    frame_e = {
        "name": "balances",
        "msg": [
            {"id": 501, "type": 1, "is_active": False},
            {"id": 502, "type": 4, "is_active": True},
        ],
    }
    extracted_e = transport._extract_active_balance_info(frame_e, source_frame="balances")
    assert extracted_e.balance_id == "502"
    assert extracted_e.account_type == AccountType.PRACTICE
    assert extracted_e.is_practice is True
    assert extracted_e.location == "balances[1].is_active"

    # Variant F: balance-changed notification
    frame_f = {
        "name": "balance-changed",
        "msg": {
            "current_balance": {
                "id": 601,
                "type": 4,
                "amount": 10000.0,
            }
        },
    }
    extracted_f = transport._extract_active_balance_info(frame_f, source_frame="balance-changed")
    assert extracted_f.balance_id == "601"
    assert extracted_f.account_type == AccountType.PRACTICE
    assert extracted_f.is_practice is True
    assert extracted_f.location == "msg.current_balance.id"


# ===========================================================================
# 2. Non-Destructive Partial Profile Update Tests
# ===========================================================================
def test_partial_profile_update_does_not_wipe_balance_id() -> None:
    """Ensure partial broker profile updates merge without wiping active balance_id."""
    transport = IQOptionLivePracticeTransport(ssid="test_ssid")
    # Seed initial profile with balance_id
    initial_raw = {
        "user_id": 777,
        "balance_id": 101,
        "balance_type": 4,
        "currency": "USD",
    }
    transport._profile = Profile.from_raw(initial_raw)
    transport._active_balance_id = "101"

    # Incoming partial frame without balance_id (e.g. settings/audio update)
    partial_frame = {
        "name": "profile",
        "msg": {
            "sfx_volume": 75,
            "client_category_id": 2,
        },
    }
    transport._handle_profile_frame(partial_frame)

    assert transport.profile is not None
    assert transport.profile.raw.get("balance_id") == 101
    assert transport.active_balance_id == "101"
    assert transport.profile.raw.get("sfx_volume") == 75


# ===========================================================================
# 3. Dynamic Balance-Changed Push Frame Tests
# ===========================================================================
def test_balance_changed_frame_updates_active_balance() -> None:
    """Broker balance-changed frame updates active balance, balance list, and marks practice."""
    transport = IQOptionLivePracticeTransport(ssid="test_ssid")
    transport._profile = Profile.from_raw({"user_id": 1, "balance_id": 200, "balance_type": 1})
    transport._balances = [
        Balance(
            id="101", type=AccountType.PRACTICE, amount=10000.0, currency="USD", is_active=False
        ),
        Balance(id="200", type=AccountType.REAL, amount=50.0, currency="USD", is_active=True),
    ]
    transport._active_balance_id = "200"

    # Receive balance-changed frame
    changed_frame = {
        "name": "balance-changed",
        "msg": {
            "current_balance": {
                "id": 101,
                "type": 4,
                "amount": 10000.0,
            }
        },
    }
    transport._handle_balance_changed_frame(changed_frame)

    assert transport.active_balance_id == "101"
    assert transport.profile is not None
    assert transport.profile.raw.get("balance_id") == 101
    assert transport._session_practice_selected is True
    # Verify balance list updated
    for b in transport.balances:
        if str(b.id) == "101":
            assert b.is_active is True
        elif str(b.id) == "200":
            assert b.is_active is False


# ===========================================================================
# 4. Delayed Propagation Tests (MockBrokerPracticeTransport)
# ===========================================================================
def test_delayed_change_balance_propagation_accepted_in_mock() -> None:
    """MockBrokerPracticeTransport accepts selection after simulated delayed propagation."""
    mock = MockBrokerPracticeTransport(
        initial_active_balance_id="200",
        practice_balance_id="101",
        virtual_amount=10000.0,
        delayed_balance_propagation_steps=2,
    )
    assert mock.active_balance_id == "200"
    assert mock.is_practice_verified is False

    bal = mock.select_practice_balance(operator_authorized=True)

    assert bal.id == "101"
    assert bal.type == AccountType.PRACTICE
    assert mock.active_balance_id == "101"
    assert mock.is_practice_verified is True
    # Exact wire frame dispatched ONCE
    assert len(mock.dispatched_frames) == 1
    assert mock.dispatched_frames[0]["msg"]["name"] == "change_balance"


# ===========================================================================
# 5. Delayed Propagation Tests (Live Practice Transport + MockWebSocket)
# ===========================================================================
class DelayedMockWebSocket(MockWebSocket):
    """Mock websocket where balance switch propagates on the 2nd poll step."""

    def __init__(
        self, initial_messages: list[str] | None = None, target_practice_id: int = 101
    ) -> None:
        super().__init__(initial_messages=initial_messages)
        self.target_practice_id = target_practice_id
        self.poll_count = 0
        self.change_balance_called = False
        self.active_balance_id = 200  # Starts as REAL

    async def _record_send(self, data: str) -> None:
        self.sent_messages.append(data)
        parsed = json.loads(data)
        msg_obj = parsed.get("msg")
        msg_name = msg_obj.get("name") if isinstance(msg_obj, dict) else None

        if msg_name == "change_balance":
            self.change_balance_called = True
            # Zero response for change_balance directly

        if msg_name == "get-profile":
            if self.change_balance_called:
                self.poll_count += 1
                if self.poll_count >= 2:
                    self.active_balance_id = self.target_practice_id

            self.inject_message({
                "name": "profile",
                "request_id": parsed.get("request_id"),
                "msg": {
                    "result": {
                        "user_id": 999,
                        "balance_id": self.active_balance_id,
                        "balance_type": 4
                        if self.active_balance_id == self.target_practice_id
                        else 1,
                    }
                },
            })

        if msg_name == "get-balances":
            is_practice = self.active_balance_id == self.target_practice_id
            self.inject_message({
                "name": "balances",
                "request_id": parsed.get("request_id"),
                "msg": [
                    {
                        "id": 101,
                        "type": 4,
                        "amount": 10000.0,
                        "currency": "USD",
                        "is_active": is_practice,
                    },
                    {
                        "id": 200,
                        "type": 1,
                        "amount": 50.0,
                        "currency": "USD",
                        "is_active": not is_practice,
                    },
                ],
            })


@pytest.mark.asyncio
async def test_delayed_propagation_in_live_transport() -> None:
    """Live transport bounded verification loop accepts delayed propagation without resending."""
    handshake = [
        json.dumps({"name": "authenticated", "msg": True}),
        json.dumps({
            "name": "profile",
            "msg": {
                "user_id": 999123,
                "balance_id": 200,
                "balance_type": 1,
            },
        }),
        json.dumps({
            "name": "balances",
            "msg": [
                {
                    "id": 101,
                    "type": 4,
                    "amount": 10000.0,
                    "currency": "USD",
                    "is_active": False,
                },
                {
                    "id": 200,
                    "type": 1,
                    "amount": 50.0,
                    "currency": "USD",
                    "is_active": True,
                },
            ],
        }),
    ]
    mock_ws = DelayedMockWebSocket(initial_messages=handshake, target_practice_id=101)

    async def fake_connect() -> MockWebSocket:
        return mock_ws

    transport = IQOptionLivePracticeTransport(
        ssid="test_mock_ssid",
        config=IQOptionLivePracticeTransportConfig(request_timeout=2.0),
        connect_factory=fake_connect,
    )

    try:
        await transport.async_connect(require_practice=False)
        assert transport.is_authenticated is True
        assert transport.active_balance_id == "200"
        assert transport.is_practice_verified is False

        # Run selection with bounded verification
        bal = await transport.async_select_practice_balance(
            operator_authorized=True,
            verification_timeout=2.0,
            retry_interval=0.05,
        )

        assert bal.id == "101"
        assert bal.type == AccountType.PRACTICE
        assert transport.active_balance_id == "101"
        assert transport.is_practice_verified is True

        # Invariant: change_balance sent EXACTLY ONCE
        change_balance_frames = [
            m
            for m in mock_ws.sent_messages
            if isinstance(json.loads(m).get("msg"), dict)
            and json.loads(m)["msg"].get("name") == "change_balance"
        ]
        assert len(change_balance_frames) == 1
    finally:
        await transport.async_close()


# ===========================================================================
# 6. Verification Timeout Tests (Fails Closed)
# ===========================================================================
def test_verification_timeout_fails_closed_in_mock() -> None:
    """Verification timeout raises PracticeAuthorizationError and active balance remains Real."""
    mock = MockBrokerPracticeTransport(
        initial_active_balance_id="200",
        practice_balance_id="101",
        virtual_amount=10000.0,
        delayed_balance_propagation_steps=999,
    )

    with pytest.raises(PracticeAuthorizationError) as exc_info:
        mock.select_practice_balance(
            operator_authorized=True,
            verification_timeout=0.1,
            retry_interval=0.02,
        )

    assert "failed to confirm Practice balance selection" in str(exc_info.value)
    assert mock.active_balance_id == "200"
    assert mock.is_practice_verified is False


# ===========================================================================
# 7. Disconnect During Verification Fails Closed
# ===========================================================================
@pytest.mark.asyncio
async def test_session_disconnect_during_verification_fails_closed() -> None:
    """Disconnect during verification loop raises ConnectionError fail-closed."""
    transport = IQOptionLivePracticeTransport(ssid="test_ssid")
    transport._ws = MockWebSocket()
    transport._session_id = "test_sess_disconnect"
    transport._connected = True

    transport._profile = Profile.from_raw({"user_id": 999, "balance_id": 200, "balance_type": 1})
    transport._balances = [
        Balance(
            id="101", type=AccountType.PRACTICE, amount=10000.0, currency="USD", is_active=False
        ),
        Balance(id="200", type=AccountType.REAL, amount=50.0, currency="USD", is_active=True),
    ]
    transport._active_balance_id = "200"

    # Simulate disconnect during async_refresh_account_state
    original_refresh = transport.async_refresh_account_state

    async def mock_refresh_disconnect(*args: Any, **kwargs: Any) -> None:
        transport._ws = None
        transport._closed = True
        await original_refresh(*args, **kwargs)

    with patch.object(
        transport, "async_refresh_account_state", side_effect=mock_refresh_disconnect
    ):
        with pytest.raises(ConnectionError) as exc_info:
            await transport.async_select_practice_balance(
                operator_authorized=True,
                verification_timeout=1.0,
                retry_interval=0.05,
            )
        assert "disconnected" in str(exc_info.value).lower()
    assert transport.is_practice_verified is False


# ===========================================================================
# 8. CLI Preflight-Only Mode Tests
# ===========================================================================
def test_preflight_only_cli_exits_cleanly_with_zero_orders() -> None:
    """--preflight-only executes all verification checks but completely bypasses order dispatch."""
    result = runner.invoke(
        app,
        [
            "practice-test",
            "--offline-mock",
            "--preflight-only",
            "--asset",
            "EURUSD",
            "--side",
            "call",
            "--stake",
            "1.00",
        ],
    )

    assert result.exit_code == 0
    assert "LIVE PRE-FLIGHT VERIFICATION SUCCEEDED: ZERO ORDERS DISPATCHED" in result.stdout
    assert "Preflight-Only Mode (Zero Orders Dispatched)" in result.stdout
    assert "Pre-Flight Mode:" in result.stdout
    assert "BYPASSED" in result.stdout
    # Invariant: prompt to transmit order is never shown
    assert "Transmit single" not in result.stdout
    # Invariant: order submission message is not in stdout
    assert "Submitting order to broker transport" not in result.stdout
    assert "Broker Ticket ID" not in result.stdout


def test_preflight_only_with_select_practice_flag() -> None:
    """--preflight-only with --select-practice selects practice and verifies preflight."""
    result = runner.invoke(
        app,
        [
            "practice-test",
            "--offline-mock",
            "--select-practice",
            "--preflight-only",
            "--asset",
            "EURUSD",
            "--side",
            "call",
            "--stake",
            "1.00",
            "--confirm",
        ],
    )

    assert result.exit_code == 0
    assert "LIVE PRE-FLIGHT VERIFICATION SUCCEEDED: ZERO ORDERS DISPATCHED" in result.stdout
    assert "PRACTICE" in result.stdout


# ===========================================================================
# 9. CLI Practice Selection Failure Clean Rich Panel Tests
# ===========================================================================
def test_practice_selection_failure_in_cli_displays_clean_panel() -> None:
    """Practice selection failure in CLI presents a clean Rich panel with zero tracebacks."""
    with patch(
        "iqoption_research.execution.practice_transport.MockBrokerPracticeTransport.select_practice_balance",
        side_effect=PracticeAuthorizationError(
            "Broker account state refresh failed to confirm Practice balance selection."
        ),
    ):
        result = runner.invoke(
            app,
            [
                "practice-test",
                "--offline-mock",
                "--select-practice",
                "--asset",
                "EURUSD",
                "--side",
                "call",
                "--stake",
                "1.00",
                "--confirm",
            ],
        )

        assert result.exit_code == 1
        assert "CRITICAL SAFETY BLOCK: Practice Balance Selection Failed" in result.stdout
        assert "ZERO ORDERS DISPATCHED (Fail-Closed)" in result.stdout
        assert "Traceback" not in result.stdout
