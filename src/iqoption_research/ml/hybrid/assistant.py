"""High-level AI Research Assistant (Feature 43).

Provides research-focused scientific critique for model evaluation, calibration,
backtest strategy analysis, and feature importance interpretation.
Strictly decoupled from trading execution: ZERO trading authority.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

from iqoption_research.ml.hybrid.base import BaseLLMProvider
from iqoption_research.ml.hybrid.mock import MockGPTProvider
from iqoption_research.ml.models.base import ModelEvaluationMetrics

RESEARCH_DISCLAIMER: Final[str] = (
    "[RESEARCH DISCLAIMER] AI Research Assistant: Statistical research critique only. "
    "Zero trading authority. Never use as automated trading instruction or financial advice."
)


class GPTResearchAssistant:
    """High-level AI research assistant for machine learning evaluation and strategy critique.

    Strictly decoupled from execution: has ZERO trading authority, cannot execute orders,
    and cannot connect to brokers.
    """

    def __init__(
        self,
        provider: BaseLLMProvider | None = None,
        include_disclaimer: bool = True,
    ) -> None:
        """Initialize research assistant. Defaults to MockGPTProvider for safe offline operation."""
        self._provider: BaseLLMProvider = provider if provider is not None else MockGPTProvider()
        self._include_disclaimer = bool(include_disclaimer)

    @property
    def provider(self) -> BaseLLMProvider:
        """Underlying LLM provider."""
        return self._provider

    @property
    def is_available(self) -> bool:
        """Check whether underlying provider is available."""
        return self._provider.is_available()

    def _format_output(self, raw_content: str) -> str:
        """Prepend mandatory research disclaimer."""
        if not self._include_disclaimer:
            return raw_content
        return f"{RESEARCH_DISCLAIMER}\n\n{raw_content}"

    def analyze_model_performance(
        self,
        metrics: dict[str, Any] | ModelEvaluationMetrics,
        model_name: str = "Model",
        train_metrics: dict[str, Any] | ModelEvaluationMetrics | None = None,
    ) -> str:
        """Generate structured scientific critique of model accuracy and calibration.

        Args:
            metrics: Out-of-sample evaluation metrics (ModelEvaluationMetrics or dict).
            model_name: Name of evaluated model.
            train_metrics: Optional in-sample training metrics to evaluate overfitting.

        Returns:
            Human-readable critique string with research disclaimer.
        """
        m_dict: dict[str, Any] = (
            metrics.to_dict() if isinstance(metrics, ModelEvaluationMetrics) else dict(metrics)
        )
        prompt_lines = [
            f"Please analyze the out-of-sample performance for model '{model_name}':",
        ]
        if "accuracy" in m_dict and m_dict["accuracy"] is not None:
            prompt_lines.append(f"- Accuracy: {float(m_dict['accuracy']):.4f}")
        if "balanced_accuracy" in m_dict and m_dict["balanced_accuracy"] is not None:
            prompt_lines.append(f"- Balanced Accuracy: {float(m_dict['balanced_accuracy']):.4f}")
        if "brier_score" in m_dict and m_dict["brier_score"] is not None:
            prompt_lines.append(f"- Brier Score: {float(m_dict['brier_score']):.4f}")
        if "log_loss" in m_dict and m_dict["log_loss"] is not None:
            prompt_lines.append(f"- Log Loss: {float(m_dict['log_loss']):.4f}")
        if "roc_auc" in m_dict and m_dict["roc_auc"] is not None:
            prompt_lines.append(f"- ROC-AUC: {float(m_dict['roc_auc']):.4f}")
        if "f1_macro" in m_dict and m_dict["f1_macro"] is not None:
            prompt_lines.append(f"- F1 Macro: {float(m_dict['f1_macro']):.4f}")
        if "sample_count" in m_dict and m_dict["sample_count"] is not None:
            prompt_lines.append(f"- Sample Count: {int(m_dict['sample_count'])}")

        kwargs: dict[str, Any] = {
            "model_name": model_name,
            "brier_score": m_dict.get("brier_score"),
            "val_accuracy": m_dict.get("accuracy"),
            "sample_count": m_dict.get("sample_count"),
        }

        if train_metrics is not None:
            t_dict = (
                train_metrics.to_dict()
                if isinstance(train_metrics, ModelEvaluationMetrics)
                else dict(train_metrics)
            )
            if "accuracy" in t_dict and t_dict["accuracy"] is not None:
                prompt_lines.append(f"- Training Accuracy: {float(t_dict['accuracy']):.4f}")
            if "brier_score" in t_dict and t_dict["brier_score"] is not None:
                prompt_lines.append(f"- Training Brier: {float(t_dict['brier_score']):.4f}")
            kwargs["train_accuracy"] = t_dict.get("accuracy")
            kwargs["train_brier"] = t_dict.get("brier_score")

        prompt = "\n".join(prompt_lines)
        response = self._provider.complete(prompt, **kwargs)
        return self._format_output(response.content)

    def critique_strategy_backtest(
        self,
        backtest_summary: dict[str, Any] | Any,
        strategy_name: str = "Strategy",
        payout_rate: float = 0.85,
    ) -> str:
        """Critique trading strategy backtest results against payoff hurdles and risk limits.

        Args:
            backtest_summary: BacktestMetrics, ResearchReport, or dict containing performance data.
            strategy_name: Name of evaluated strategy.
            payout_rate: Binary option payout rate or reference payoff.

        Returns:
            Human-readable strategy critique string.
        """
        if hasattr(backtest_summary, "metrics"):
            m_dict = backtest_summary.metrics.to_dict()
        elif hasattr(backtest_summary, "to_dict"):
            m_dict = backtest_summary.to_dict()
        else:
            m_dict = dict(backtest_summary)

        prompt_lines = [
            f"Please critique the backtest performance for strategy '{strategy_name}':",
        ]
        if "total_trades" in m_dict and m_dict["total_trades"] is not None:
            prompt_lines.append(f"- Total Trades: {int(m_dict['total_trades'])}")
        if "win_rate" in m_dict and m_dict["win_rate"] is not None:
            prompt_lines.append(f"- Win Rate: {float(m_dict['win_rate']):.4f}")
        if "net_pnl" in m_dict and m_dict["net_pnl"] is not None:
            prompt_lines.append(f"- Net PnL: {float(m_dict['net_pnl']):.2f}")
        if "profit_factor" in m_dict and m_dict["profit_factor"] is not None:
            prompt_lines.append(f"- Profit Factor: {float(m_dict['profit_factor']):.2f}")
        if "max_drawdown_percent" in m_dict and m_dict["max_drawdown_percent"] is not None:
            prompt_lines.append(f"- Max Drawdown %: {float(m_dict['max_drawdown_percent']):.2f}")
        prompt_lines.append(f"- Payout Rate: {payout_rate:.2f}")

        kwargs: dict[str, Any] = {
            "strategy_name": strategy_name,
            "total_trades": m_dict.get("total_trades"),
            "win_rate": m_dict.get("win_rate"),
            "net_pnl": m_dict.get("net_pnl"),
            "profit_factor": m_dict.get("profit_factor"),
            "max_drawdown_percent": m_dict.get("max_drawdown_percent"),
            "payout_rate": payout_rate,
        }

        prompt = "\n".join(prompt_lines)
        response = self._provider.complete(prompt, **kwargs)
        return self._format_output(response.content)

    def explain_feature_importance(
        self,
        importances: dict[str, float] | Sequence[tuple[str, float]],
        model_name: str = "Model",
    ) -> str:
        """Explain feature importance rankings and detect feature concentration risks.

        Args:
            importances: Dictionary or sequence of (feature_name, importance) pairs.
            model_name: Name of evaluated model.

        Returns:
            Human-readable explanation of feature contributions.
        """
        imp_dict: dict[str, float] = dict(importances)

        sorted_feats = sorted(imp_dict.items(), key=lambda x: float(x[1]), reverse=True)

        prompt_lines = [
            f"Explain the feature importance distribution for model '{model_name}':",
        ]
        for name, score in sorted_feats[:10]:
            prompt_lines.append(f"- {name}: {score:.4f}")

        kwargs: dict[str, Any] = {
            "model_name": model_name,
            "importances": imp_dict,
        }
        prompt = "\n".join(prompt_lines)
        response = self._provider.complete(prompt, **kwargs)

        return self._format_output(response.content)

    def generate_critique(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """General research critique generation querying underlying provider."""
        kwargs = dict(context or {})
        response = self._provider.complete(prompt, **kwargs)
        return self._format_output(response.content)
