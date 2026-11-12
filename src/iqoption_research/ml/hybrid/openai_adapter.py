"""Optional external OpenAI adapter with strict offline security guardrails (Feature 42).

SECURITY MANDATE:
1. Disabled by default.
2. Requires both `enabled=True` in constructor AND environment variable
   `IQOPTION_ENABLE_OPENAI=1`.
3. If disabled, `is_available()` returns False, and calling `complete()`
   raises `LLMSecurityError` immediately with ZERO network socket calls.
4. Never makes real network requests under demo or automated test runs.
"""

from __future__ import annotations

import os
from typing import Any

from iqoption_research.ml.hybrid.base import (
    BaseLLMProvider,
    LLMError,
    LLMProviderUnavailableError,
    LLMResponse,
    LLMSecurityError,
)


class OpenAIAssistant(BaseLLMProvider):
    """Optional external OpenAI adapter with strict offline security guardrails."""

    DEFAULT_PROVIDER_NAME: str = "openai"
    DEFAULT_MODEL_NAME: str = "gpt-4o-mini"
    ENV_ENABLE_FLAG: str = "IQOPTION_ENABLE_OPENAI"
    ENV_API_KEY: str = "OPENAI_API_KEY"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL_NAME,
        enabled: bool = False,
        provider_name: str = DEFAULT_PROVIDER_NAME,
        env_flag_name: str = ENV_ENABLE_FLAG,
        env_key_name: str = ENV_API_KEY,
    ) -> None:
        """Initialize OpenAIAssistant.

        Args:
            api_key: Optional explicit API key.
            model: Target OpenAI model.
            enabled: Explicit constructor enablement flag (must be True alongside env var).
            provider_name: Identifier name for provider.
            env_flag_name: Environment variable name for enablement flag.
            env_key_name: Environment variable name for API key.
        """
        self._provider_name = provider_name
        self._model = model
        self._enabled_flag = bool(enabled)
        self._env_flag_name = env_flag_name
        self._env_key_name = env_key_name
        self._api_key = api_key

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def default_model(self) -> str:
        return self._model

    def _is_enabled_by_env(self) -> bool:
        """Check if environment variable explicitly enables external OpenAI calls."""
        val = os.environ.get(self._env_flag_name, "0").strip().lower()
        return val in ("1", "true", "yes", "enabled")

    def _get_api_key(self) -> str | None:
        """Resolve API key from instance or environment."""
        if self._api_key and self._api_key.strip():
            return self._api_key.strip()
        env_val = os.environ.get(self._env_key_name, "").strip()
        return env_val if env_val else None

    def is_available(self) -> bool:
        """Check if provider is permitted and configured.

        Must return False unless enabled flag is set, environment flag is 1,
        and API key is present. NEVER performs network checks.
        """
        if not self._enabled_flag:
            return False
        if not self._is_enabled_by_env():
            return False
        key = self._get_api_key()
        return bool(key)

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate critique via OpenAI API if explicitly authorized.

        Raises:
            LLMSecurityError: If invoked while disabled (default state).
            LLMProviderUnavailableError: If enabled but API key is missing or openai not installed.
            LLMError: If prompt is empty or API call fails.
        """
        del kwargs
        # Guard 1: Offline Security Check
        if not self._enabled_flag or not self._is_enabled_by_env():
            raise LLMSecurityError(
                "OpenAIAssistant is disabled for offline safety. "
                "To enable real API calls in an authorized environment, pass enabled=True "
                f"and set environment variable {self._env_flag_name}=1."
            )

        # Guard 2: API Key Check
        key = self._get_api_key()
        if not key:
            raise LLMProviderUnavailableError(
                f"OpenAI API key missing. Set environment variable {self._env_key_name} "
                "or pass api_key to OpenAIAssistant."
            )

        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            raise LLMError("Prompt must be a non-empty string.")

        # Guard 3: Lazy import openai package
        try:
            import openai
        except ImportError as e:
            raise LLMProviderUnavailableError(
                "Package 'openai' is not installed. Install via 'pip install openai'."
            ) from e

        # Authorized API Execution
        target_model = model or self._model
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            client = openai.OpenAI(api_key=key)
            completion = client.chat.completions.create(
                model=target_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )
            choice = completion.choices[0]
            content = choice.message.content or ""
            tokens = completion.usage.total_tokens if completion.usage else None

            return LLMResponse(
                content=content,
                provider=self.provider_name,
                model=target_model,
                tokens_used=tokens,
                finish_reason=choice.finish_reason,
                metadata={"id": completion.id, "created": completion.created},
            )
        except Exception as e:
            raise LLMError(f"OpenAI API completion failed: {e}") from e
