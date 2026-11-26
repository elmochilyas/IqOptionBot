"""Tests for secure HTTP authentication request construction and responses."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from iqoption_research.connectors.iqoption.exceptions import (
    ConnectionError as IQConnectionError,
)
from iqoption_research.connectors.iqoption.exceptions import (
    LoginError,
    RequestTimeoutError,
    TwoFactorRequiredError,
)
from iqoption_research.connectors.iqoption.http_auth import async_login, async_logout


@pytest.mark.asyncio
async def test_async_login_success(mock_credentials):
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"code": "success"})
    cookie_mock = MagicMock()
    cookie_mock.value = "mock_ssid_12345"
    mock_response.cookies = {"ssid": cookie_mock}

    # Context manager setup for `async with response:`
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = AsyncMock(return_value=mock_response)
    mock_session.close = AsyncMock()

    with patch("aiohttp.ClientSession", return_value=mock_session):
        ssid = await async_login(mock_credentials)
        assert ssid == "mock_ssid_12345"

        # Verify POST arguments
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert call_args[0][0] == "https://auth.iqoption.com/api/v2/login"
        assert call_args[1]["data"]["identifier"] == mock_credentials.email
        assert call_args[1]["data"]["password"] == mock_credentials.password


@pytest.mark.asyncio
async def test_async_login_invalid_credentials(mock_credentials):
    mock_response = MagicMock()
    mock_response.status = 401
    mock_response.json = AsyncMock(return_value={"code": "invalid_credentials"})
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = AsyncMock(return_value=mock_response)
    mock_session.close = AsyncMock()

    with (
        patch("aiohttp.ClientSession", return_value=mock_session),
        pytest.raises(LoginError, match="HTTP 401"),
    ):
        await async_login(mock_credentials)


@pytest.mark.asyncio
async def test_async_login_2fa_challenge_rejected(mock_credentials):
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.cookies = {}
    mock_response.json = AsyncMock(return_value={"code": "verify", "message": "2FA required"})
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = AsyncMock(return_value=mock_response)
    mock_session.close = AsyncMock()

    with (
        patch("aiohttp.ClientSession", return_value=mock_session),
        pytest.raises(TwoFactorRequiredError) as exc_info,
    ):
        await async_login(mock_credentials)
    # Must fail safely and must NEVER instruct user to disable 2FA
    assert "Two-factor authentication" in str(exc_info.value)
    assert "disable" not in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_async_login_endpoint_validation_rejects_unauthorized_url(mock_credentials):
    with pytest.raises(ValueError, match="not allowed"):
        await async_login(mock_credentials, login_url="https://attacker.com/login")


@pytest.mark.asyncio
async def test_async_login_timeout(mock_credentials):
    mock_session = MagicMock()
    mock_session.post = AsyncMock(side_effect=TimeoutError("Request timed out"))
    mock_session.close = AsyncMock()

    with (
        patch("aiohttp.ClientSession", return_value=mock_session),
        pytest.raises(RequestTimeoutError, match="timed out"),
    ):
        await async_login(mock_credentials)


@pytest.mark.asyncio
async def test_async_login_client_error(mock_credentials):
    mock_session = MagicMock()
    mock_session.post = AsyncMock(side_effect=aiohttp.ClientConnectionError("DNS failure"))
    mock_session.close = AsyncMock()

    with (
        patch("aiohttp.ClientSession", return_value=mock_session),
        pytest.raises(IQConnectionError, match="request failed"),
    ):
        await async_login(mock_credentials)


@pytest.mark.asyncio
async def test_async_logout(mock_credentials):
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = AsyncMock(return_value=mock_response)
    mock_session.close = AsyncMock()

    with patch("aiohttp.ClientSession", return_value=mock_session):
        success = await async_logout("dummy_ssid")
        assert success is True
        assert mock_session.post.call_args[1]["allow_redirects"] is False


@pytest.mark.parametrize("status_code", [301, 302, 307, 308])
@pytest.mark.asyncio
async def test_async_login_rejects_redirects(mock_credentials: Any, status_code: int) -> None:
    mock_response = MagicMock()
    mock_response.status = status_code
    mock_response.json = AsyncMock(return_value={})
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = AsyncMock(return_value=mock_response)
    mock_session.close = AsyncMock()

    with (
        patch("aiohttp.ClientSession", return_value=mock_session),
        pytest.raises(LoginError, match="redirect rejected"),
    ):
        await async_login(mock_credentials)

    assert mock_session.post.call_args[1]["allow_redirects"] is False


@pytest.mark.parametrize("status_code", [301, 302, 307, 308])
@pytest.mark.asyncio
async def test_async_logout_rejects_redirects(status_code: int) -> None:
    mock_response = MagicMock()
    mock_response.status = status_code
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = AsyncMock(return_value=mock_response)
    mock_session.close = AsyncMock()

    with patch("aiohttp.ClientSession", return_value=mock_session):
        success = await async_logout("dummy_ssid")
        assert success is False
        assert mock_session.post.call_args[1]["allow_redirects"] is False
