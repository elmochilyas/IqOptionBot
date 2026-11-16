"""OANDA v20 Practice REST API client (Phase 18).

Provides asynchronous HTTP client for OANDA v20 Practice execution:
- Strict Practice-only endpoint allowlist (https://api-fxpractice.oanda.com/v3)
- Permanent hard block against Real/Production endpoints (api-fxtrade.oanda.com)
- Strict TLS 1.2+ certificate and hostname verification via certifi
- Fail-closed error handling with typed exception taxonomy
- Full redaction of bearer tokens from logs and error messages
"""

from __future__ import annotations

from typing import Any, Final

import aiohttp

from iqoption_research.execution.oanda.models import (
    OandaAccountSummary,
    OandaInstrumentDetails,
    OandaMarketOrderRequest,
    OandaOrderFillTransaction,
    OandaPrice,
    OandaSubmissionResponse,
    OandaTradeSummary,
)
from iqoption_research.execution.safety import RealAccountExecutionBlockedError
from iqoption_research.security.tls import create_tls_context

OANDA_PRACTICE_BASE_URL: Final[str] = "https://api-fxpractice.oanda.com/v3"
OANDA_PROHIBITED_REAL_URL: Final[str] = "https://api-fxtrade.oanda.com/v3"


class OandaClientError(RuntimeError):
    """Base exception for all OANDA client operations."""


class OandaAuthenticationError(OandaClientError):
    """Raised when OANDA credentials or token are rejected (HTTP 401)."""


class OandaApiError(OandaClientError):
    """Raised when OANDA returns an HTTP 4xx or 5xx error payload."""

    def __init__(self, status_code: int, message: str, error_code: str | None = None) -> None:
        super().__init__(f"OANDA API error {status_code} ({error_code or 'UNKNOWN'}): {message}")
        self.status_code = status_code
        self.error_message = message
        self.error_code = error_code


class OandaTimeoutError(OandaClientError):
    """Raised when an OANDA request times out."""


class OandaConnectionError(OandaClientError):
    """Raised when network transport fails or connection is dropped."""


class OandaPracticeClient:
    """Guarded asynchronous client for the OANDA v20 Practice API."""

    def __init__(
        self,
        token: str,
        *,
        base_url: str = OANDA_PRACTICE_BASE_URL,
        request_timeout_seconds: float = 10.0,
    ) -> None:
        if not token or not token.strip():
            raise ValueError("OANDA practice token must not be empty")

        # Security enforcement: verify endpoint is strictly practice
        if "api-fxtrade.oanda.com" in base_url or "stream-fxtrade.oanda.com" in base_url:
            raise RealAccountExecutionBlockedError(
                "CRITICAL SECURITY ENFORCEMENT: OANDA Real/Production endpoints "
                "are permanently prohibited. Only Practice endpoints are allowed."
            )
        if not base_url.startswith(OANDA_PRACTICE_BASE_URL):
            raise ValueError(
                f"Invalid OANDA base URL '{base_url}'. Expected '{OANDA_PRACTICE_BASE_URL}'"
            )

        self._token = token.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = request_timeout_seconds
        self._session: aiohttp.ClientSession | None = None

    @property
    def is_closed(self) -> bool:
        return self._session is None or self._session.closed

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            tls_ctx = create_tls_context()
            connector = aiohttp.TCPConnector(ssl=tls_ctx)
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "IqOptionBot-Research/1.0",
                },
            )
        return self._session

    async def close(self) -> None:
        """Close underlying client session and connections."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Perform authenticated HTTPS request against OANDA practice API."""
        url = f"{self.base_url}/{path.lstrip('/')}"
        session = await self._get_session()

        try:
            async with session.request(method, url, json=json_data, params=params) as response:
                status = response.status
                try:
                    payload = await response.json()
                except Exception:  # noqa: BLE001
                    text_body = await response.text()
                    payload = {"errorMessage": text_body}

                if status == 200 or status == 201:
                    return payload if isinstance(payload, dict) else {}

                error_msg = str(
                    payload.get("errorMessage") or payload.get("message") or f"HTTP {status}"
                )
                error_code = str(payload.get("errorCode") or "")

                if status == 401:
                    raise OandaAuthenticationError("OANDA token authentication failed (401)")
                raise OandaApiError(status, error_msg, error_code or None)

        except TimeoutError as exc:
            raise OandaTimeoutError(
                f"OANDA request to {path} timed out after {self.timeout_seconds}s"
            ) from exc
        except aiohttp.ClientConnectorError as exc:
            raise OandaConnectionError(
                f"Failed to connect to OANDA practice endpoint: {exc}"
            ) from exc
        except (OandaClientError, RealAccountExecutionBlockedError):
            raise
        except Exception as exc:
            raise OandaConnectionError(f"Transport error connecting to OANDA: {exc}") from exc

    async def get_account_summary(self, account_id: str) -> OandaAccountSummary:
        """Retrieve authoritative account balance and margin details."""
        data = await self._request("GET", f"accounts/{account_id}/summary")
        acc_dict = data.get("account")
        if not isinstance(acc_dict, dict):
            raise OandaApiError(200, "Malformed response: missing 'account' dict")
        return OandaAccountSummary.model_validate(acc_dict)

    async def get_instrument_details(
        self, account_id: str, instrument: str
    ) -> OandaInstrumentDetails:
        """Fetch specification and pip precision for an instrument."""
        data = await self._request(
            "GET",
            f"accounts/{account_id}/instruments",
            params={"instruments": instrument},
        )
        instruments = data.get("instruments", [])
        if not instruments or not isinstance(instruments, list):
            raise OandaApiError(404, f"Instrument {instrument} not found on account")
        return OandaInstrumentDetails.model_validate(instruments[0])

    async def get_pricing(self, account_id: str, instruments: list[str]) -> list[OandaPrice]:
        """Fetch real-time executable prices and spreads."""
        data = await self._request(
            "GET",
            f"accounts/{account_id}/pricing",
            params={"instruments": ",".join(instruments)},
        )
        raw_prices = data.get("prices", [])
        return [OandaPrice.model_validate(p) for p in raw_prices]

    async def submit_market_order(
        self, account_id: str, request: OandaMarketOrderRequest
    ) -> OandaSubmissionResponse:
        """Submit a market order to the OANDA Practice account."""
        data = await self._request(
            "POST",
            f"accounts/{account_id}/orders",
            json_data=request.model_dump(exclude_none=True),
        )
        return OandaSubmissionResponse.model_validate(data)

    async def get_trade(self, account_id: str, trade_id: str) -> OandaTradeSummary:
        """Query state and PnL of an individual trade."""
        data = await self._request("GET", f"accounts/{account_id}/trades/{trade_id}")
        trade_dict = data.get("trade")
        if not isinstance(trade_dict, dict):
            raise OandaApiError(404, f"Trade {trade_id} not found")
        return OandaTradeSummary.model_validate(trade_dict)

    async def close_trade(
        self, account_id: str, trade_id: str, units: str | None = None
    ) -> OandaOrderFillTransaction:
        """Close an open trade and retrieve authoritative settlement transaction."""
        body = {"units": units} if units is not None else {}
        data = await self._request(
            "PUT",
            f"accounts/{account_id}/trades/{trade_id}/close",
            json_data=body,
        )
        fill = data.get("orderFillTransaction")
        if not isinstance(fill, dict):
            raise OandaApiError(200, "Trade close response missing orderFillTransaction")
        return OandaOrderFillTransaction.model_validate(fill)
