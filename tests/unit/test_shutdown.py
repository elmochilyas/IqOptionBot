"""Tests for graceful shutdown and resource cleanup."""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.connectors.simulation import SimulatedReadOnlyConnector
from iqoption_research.models.connection import ConnectionState, DisconnectReason


@pytest.mark.asyncio
async def test_async_iqoption_close_idempotent(mock_credentials: Any) -> None:
    client = AsyncIQOption(mock_credentials, offline=True)
    # Never connected; close should be safe and idempotent
    await client.close()
    await client.close()
    assert client.state == ConnectionState.DISCONNECTED


@pytest.mark.asyncio
async def test_async_iqoption_close_cleans_ws_and_session(mock_credentials: Any) -> None:
    client = AsyncIQOption(mock_credentials, offline=True)
    mock_ws = AsyncMock()
    mock_ws.close = AsyncMock()
    mock_ws.state = ConnectionState.READY

    client._ws = mock_ws
    client._ssid = "active_session_ssid"

    await client.close(reason=DisconnectReason.USER_REQUESTED)
    mock_ws.close.assert_called_once_with(DisconnectReason.USER_REQUESTED)
    assert client._ws is None
    assert client._ssid is None
    assert client._credentials is None


@pytest.mark.asyncio
async def test_simulated_connector_shutdown() -> None:
    connector = SimulatedReadOnlyConnector()
    await connector.connect()
    assert connector.state is ConnectionState.READY

    await connector.close()
    final_state: ConnectionState = connector.state
    assert final_state is ConnectionState.DISCONNECTED
    # Repeated close is idempotent
    await connector.close()


@pytest.mark.asyncio
async def test_async_iqoption_connect_ws_failure_triggers_session_logout(
    mock_credentials: Any,
) -> None:
    from unittest.mock import patch

    client = AsyncIQOption(mock_credentials, offline=False)

    with patch(
        "iqoption_research.connectors.iqoption.client.async_login",
        new=AsyncMock(return_value="valid_authenticated_ssid"),
    ), patch(
        "iqoption_research.connectors.iqoption.client.IQOptionWebSocketClient.connect",
        new=AsyncMock(side_effect=ConnectionError("WS simulated connect failure")),
    ), patch(
        "iqoption_research.connectors.iqoption.client.async_logout",
        new=AsyncMock(return_value=True),
    ) as mock_logout:
        with pytest.raises(ConnectionError, match="WS simulated connect failure"):
            await client.connect()

        # Session cleanup: must perform best-effort logout with valid ssid
        mock_logout.assert_called_once()
        assert mock_logout.call_args[0][0] == "valid_authenticated_ssid"
        assert client._ssid is None
        assert client.state == ConnectionState.FAILED
