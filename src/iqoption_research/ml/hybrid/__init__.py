"""Hybrid Intelligence & LLM Research Assistant Subsystem.

Provides abstract provider protocols, deterministic offline mock critiquer,
security-guarded OpenAI adapter (disabled by default), and high-level
research assistant for model diagnostics and strategy backtest evaluation.
"""

from __future__ import annotations

from iqoption_research.ml.hybrid.assistant import (
    RESEARCH_DISCLAIMER,
    GPTResearchAssistant,
)
from iqoption_research.ml.hybrid.base import (
    BaseLLMProvider,
    LLMError,
    LLMProviderUnavailableError,
    LLMResponse,
    LLMSecurityError,
)
from iqoption_research.ml.hybrid.mock import MockGPTProvider
from iqoption_research.ml.hybrid.openai_adapter import OpenAIAssistant

__all__ = [
    "RESEARCH_DISCLAIMER",
    "BaseLLMProvider",
    "GPTResearchAssistant",
    "LLMError",
    "LLMProviderUnavailableError",
    "LLMResponse",
    "LLMSecurityError",
    "MockGPTProvider",
    "OpenAIAssistant",
]
