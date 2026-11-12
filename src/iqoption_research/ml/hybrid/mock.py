"""Deterministic offline mock GPT provider (Feature 41).

Zero network I/O, deterministic heuristics for scientific research critique,
overfitting detection, Brier calibration evaluation, break-even hurdle analysis,
drawdown warnings, and feature concentration detection.
"""

from __future__ import annotations

import re
from typing import Any

from iqoption_research.ml.hybrid.base import (
    BaseLLMProvider,
    LLMError,
    LLMResponse,
)


class MockGPTProvider(BaseLLMProvider):
    """Deterministic, fully offline research critiquer requiring zero network I/O.

    Generates structured scientific critiques based on prompt keyword parsing
    and numerical metrics (overfitting detection, Brier score analysis,
    calibration status, win rate vs break-even hurdle, drawdown warnings,
    and feature importance concentration).
    """

    DEFAULT_PROVIDER_NAME: str = "mock-gpt"
    DEFAULT_MODEL_NAME: str = "mock-gpt-4o-research"

    def __init__(
        self,
        provider_name: str = DEFAULT_PROVIDER_NAME,
        model: str = DEFAULT_MODEL_NAME,
        custom_responses: dict[str, str] | None = None,
    ) -> None:
        """Initialize MockGPTProvider.

        Args:
            provider_name: Identifier name for mock provider.
            model: Default model identifier.
            custom_responses: Optional dictionary mapping keyword substrings to canned responses.
        """
        self._provider_name = provider_name
        self._model = model
        self._custom_responses = dict(custom_responses or {})

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def default_model(self) -> str:
        return self._model

    def is_available(self) -> bool:
        """Mock provider is always available offline."""
        return True

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
        """Generate deterministic critique based on prompt content and metrics.

        Args:
            prompt: User/research input prompt.
            system_prompt: Optional system prompt.
            model: Optional model override.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in generated completion.
            **kwargs: Metrics and evaluation parameters.

        Returns:
            LLMResponse instance.

        Raises:
            LLMError: If prompt is empty or whitespace.
        """
        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            raise LLMError("Prompt must be a non-empty string.")

        del system_prompt

        # Check for custom response overrides
        for key, resp in self._custom_responses.items():
            if key in prompt:
                return LLMResponse(
                    content=resp,
                    provider=self.provider_name,
                    model=model or self._model,
                    tokens_used=len(resp.split()),
                    finish_reason="stop",
                    metadata={"source": "custom_override", "matched_key": key},
                )

        # Synthesize rule-based deterministic response
        content = self._synthesize_critique(prompt, kwargs)
        tokens_est = len(content.split())

        return LLMResponse(
            content=content,
            provider=self.provider_name,
            model=model or self._model,
            tokens_used=tokens_est,
            finish_reason="stop",
            metadata={
                "source": "deterministic_heuristics",
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )

    def _synthesize_critique(self, prompt: str, kwargs: dict[str, Any]) -> str:
        """Internal rule-based engine parsing metrics and formulating structured critique."""
        target = kwargs.get("model_name") or kwargs.get("strategy_name")
        if not target:
            m = re.search(r"(?:model|strategy)\s+['\"]([^'\"]+)['\"]", prompt, re.IGNORECASE)
            if m:
                target = m.group(1)

        header = (
            f"### AI Research Assistant: Statistical Performance Critique ({target})"
            if target
            else "### AI Research Assistant: Statistical Performance Critique"
        )
        lines: list[str] = [
            header,
            "",
        ]

        lower_prompt = prompt.lower()

        # Heuristic 1: Overfitting Diagnostics
        train_acc = kwargs.get("train_accuracy")
        val_acc = (
            kwargs.get("val_accuracy")
            if kwargs.get("val_accuracy") is not None
            else kwargs.get("test_accuracy")
        )
        if train_acc is not None and val_acc is not None:
            gap = float(train_acc) - float(val_acc)
            if gap > 0.10:
                lines.append(
                    f"- **Overfitting Warning**: Training accuracy ({train_acc:.2%}) "
                    f"significantly outperforms validation accuracy ({val_acc:.2%}) by {gap:.2%}. "
                    "Recommend regularization (e.g. tree depth caps, L2 regularization, "
                    "or dropping redundant features)."
                )
            else:
                lines.append(
                    f"- **Generalization Health**: Train-val accuracy gap is small ({gap:.2%}), "
                    "indicating stable generalization."
                )

        # Heuristic 2: Calibration & Brier Score
        brier = kwargs.get("brier_score")
        if brier is None and "brier" in lower_prompt:
            brier = self._extract_float_after_token(lower_prompt, "brier")

        if brier is not None:
            brier_val = float(brier)
            if brier_val > 0.25:
                lines.append(
                    f"- **Calibration Alert**: Brier score ({brier_val:.4f}) is worse than an "
                    "uninformative baseline (0.2500). Probabilities are uncalibrated; "
                    "post-hoc calibration via Platt Scaling or Isotonic Regression is required."
                )
            elif brier_val < 0.22:
                lines.append(
                    f"- **Probability Calibration**: Strong Brier score ({brier_val:.4f}) "
                    "indicates good probabilistic discrimination."
                )
            else:
                lines.append(
                    f"- **Probability Calibration**: Brier score ({brier_val:.4f}) is moderate. "
                    "Evaluate ECE reliability curves."
                )

        # Heuristic 3: Expected Value & Binary Break-Even Hurdle
        win_rate = kwargs.get("win_rate")
        payout_rate = kwargs.get("payout_rate", 0.85)
        if win_rate is not None:
            wr_val = float(win_rate)
            be_hurdle = 1.0 / (1.0 + float(payout_rate))
            if wr_val < be_hurdle:
                lines.append(
                    f"- **Negative Expectancy Warning**: Simulated win rate ({wr_val:.2%}) falls "
                    f"below theoretical break-even ({be_hurdle:.2%}) for a {payout_rate:.0%} "
                    "payout asset. Strategy has negative mathematical edge."
                )
            else:
                margin = wr_val - be_hurdle
                lines.append(
                    f"- **Positive Edge**: Win rate ({wr_val:.2%}) exceeds break-even hurdle "
                    f"({be_hurdle:.2%}) by +{margin:.2%}. Profitable subject to transaction "
                    "costs and execution friction."
                )

        # Heuristic 4: Risk & Drawdown
        max_dd = (
            kwargs.get("max_drawdown_percent")
            if kwargs.get("max_drawdown_percent") is not None
            else kwargs.get("max_drawdown")
        )
        if max_dd is not None:
            dd_val = float(max_dd)
            dd_pct = dd_val if dd_val <= 1.0 else dd_val / 100.0
            if dd_pct > 0.20:
                lines.append(
                    f"- **Drawdown Warning**: Maximum drawdown ({dd_pct:.1%}) exceeds conservative "
                    "20% risk ceiling. Review position sizing or implement trade abstention."
                )
            else:
                lines.append(
                    f"- **Drawdown Profile**: Maximum drawdown ({dd_pct:.1%}) remains "
                    "within acceptable limits."
                )

        # Heuristic 5: Sample Size Sufficiency
        trade_count = (
            kwargs.get("total_trades")
            if kwargs.get("total_trades") is not None
            else kwargs.get("sample_count")
        )
        if trade_count is not None:
            n_val = int(trade_count)
            if n_val < 100:
                lines.append(
                    f"- **Sample Size Warning**: Sample size (N={n_val}) is statistically low. "
                    "Backtest results have wide confidence intervals; minimum 100+ trades advised."
                )

        # Heuristic 6: Feature Importances Review
        importances = kwargs.get("importances") or kwargs.get("feature_importances")
        if isinstance(importances, dict) and importances:
            sorted_feats = sorted(importances.items(), key=lambda x: float(x[1]), reverse=True)
            top_3 = sorted_feats[:3]
            top_str = ", ".join(f"'{k}' ({float(v):.3f})" for k, v in top_3)
            lines.append(f"- **Key Feature Drivers**: Top predictors are {top_str}.")

            total_imp = sum(float(v) for v in importances.values())
            if total_imp > 0 and (float(top_3[0][1]) / total_imp) > 0.50:
                lines.append(
                    f"- **Concentration Risk**: Top feature '{top_3[0][0]}' accounts for over "
                    "50% of model importance. Check for collinearity or target leakage."
                )

        # Default scientific summary if no specific metrics were found
        if len(lines) == 2:
            lines.append(
                "- **Analysis Summary**: Model and strategy components evaluated under offline "
                "research framework. Ensure strict chronological walk-forward validation and "
                "verify probability calibration before live deployment."
            )

        lines.append("")
        lines.append("**Recommendations**:")
        lines.append(
            "1. Validate that out-of-sample Brier score improves upon individual base models."
        )
        lines.append(
            "2. Confirm that trade abstention filters out low-conviction signals near break-even."
        )
        lines.append("3. Verify that feature importances do not exhibit single-predictor leakage.")

        return "\n".join(lines)

    @staticmethod
    def _extract_float_after_token(text: str, token: str) -> float | None:
        """Helper to extract float following a keyword in prompt text."""
        parts = text.split(token)
        if len(parts) > 1:
            words = parts[1].replace(":", " ").replace("=", " ").split()
            for w in words:
                try:
                    return float(w.rstrip(",;%"))
                except ValueError:
                    continue
        return None
