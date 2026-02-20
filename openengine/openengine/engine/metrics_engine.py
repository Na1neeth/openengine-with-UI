"""
Metrics Engine — computes all robustness metrics from backtest results.

Takes an equity curve (DataFrame) and a list of Trade objects and returns
a structured dict of performance metrics.
"""

import math
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from .models import Trade, Direction


class MetricsEngine:
    """Computes comprehensive backtest performance metrics."""

    @staticmethod
    def compute(
        equity_curve: pd.DataFrame,
        trades: List[Trade],
        drawdown_series: pd.Series,
        initial_capital: float,
        risk_free_rate: float = 0.0,
    ) -> Dict[str, Any]:
        """Compute all metrics and return as a structured dict.

        Args:
            equity_curve: DataFrame with 'total' column (indexed by date).
            trades: list of completed Trade objects.
            drawdown_series: Series of drawdown fractions (negative values).
            initial_capital: starting capital.
            risk_free_rate: annual risk-free rate for Sharpe/Sortino (default 0).

        Returns:
            Dict with all metrics, ready for JSON serialisation.
        """
        metrics: Dict[str, Any] = {}

        total = equity_curve["total"] if "total" in equity_curve.columns else pd.Series(dtype=float)

        # --- Capital metrics ---
        final_value = float(total.iloc[-1]) if len(total) > 0 else initial_capital
        total_return = ((final_value - initial_capital) / initial_capital) * 100 if initial_capital > 0 else 0

        metrics["initial_capital"] = round(initial_capital, 2)
        metrics["final_value"] = round(final_value, 2)
        metrics["total_return_pct"] = round(total_return, 2)

        # --- CAGR ---
        metrics["cagr_pct"] = round(MetricsEngine._cagr(initial_capital, final_value, total), 2)

        # --- Drawdown ---
        max_dd = abs(float(drawdown_series.min())) * 100 if len(drawdown_series) > 0 else 0
        metrics["max_drawdown_pct"] = round(max_dd, 2)

        # Drawdown duration
        max_dd_duration = MetricsEngine._max_dd_duration(drawdown_series)
        metrics["max_drawdown_duration_bars"] = max_dd_duration

        # --- Trade statistics ---
        n_trades = len(trades)
        metrics["total_trades"] = n_trades

        if n_trades > 0:
            winners = [t for t in trades if t.net_pnl > 0]
            losers = [t for t in trades if t.net_pnl <= 0]

            win_rate = (len(winners) / n_trades) * 100
            avg_win = np.mean([t.net_pnl for t in winners]) if winners else 0
            avg_loss = abs(np.mean([t.net_pnl for t in losers])) if losers else 0

            gross_profit = sum(t.net_pnl for t in winners) if winners else 0
            gross_loss = abs(sum(t.net_pnl for t in losers)) if losers else 0

            metrics["winning_trades"] = len(winners)
            metrics["losing_trades"] = len(losers)
            metrics["win_rate_pct"] = round(win_rate, 2)
            metrics["avg_win"] = round(float(avg_win), 2)
            metrics["avg_loss"] = round(float(avg_loss), 2)
            metrics["risk_reward_ratio"] = round(float(avg_win / avg_loss), 2) if avg_loss > 0 else 0
            metrics["expectancy"] = round(
                (win_rate / 100 * float(avg_win)) - ((1 - win_rate / 100) * float(avg_loss)), 2
            )
            metrics["profit_factor"] = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")

            # Total fees
            metrics["total_fees"] = round(sum(t.fees for t in trades), 2)

            # Average holding period
            metrics["avg_holding_period"] = round(np.mean([t.holding_period for t in trades]), 1)

            # Largest win / loss
            metrics["largest_win"] = round(max(t.net_pnl for t in trades), 2)
            metrics["largest_loss"] = round(min(t.net_pnl for t in trades), 2)
        else:
            metrics["winning_trades"] = 0
            metrics["losing_trades"] = 0
            metrics["win_rate_pct"] = 0
            metrics["avg_win"] = 0
            metrics["avg_loss"] = 0
            metrics["risk_reward_ratio"] = 0
            metrics["expectancy"] = 0
            metrics["profit_factor"] = 0
            metrics["total_fees"] = 0
            metrics["avg_holding_period"] = 0
            metrics["largest_win"] = 0
            metrics["largest_loss"] = 0

        # --- Risk-adjusted returns ---
        returns = total.pct_change().dropna() if len(total) > 1 else pd.Series(dtype=float)

        metrics["sharpe_ratio"] = round(MetricsEngine._sharpe(returns, risk_free_rate), 2)
        metrics["sortino_ratio"] = round(MetricsEngine._sortino(returns, risk_free_rate), 2)

        return metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cagr(initial: float, final: float, total_series: pd.Series) -> float:
        """Compound Annual Growth Rate."""
        if initial <= 0 or final <= 0 or len(total_series) < 2:
            return 0.0

        idx = total_series.index
        start = idx[0]
        end = idx[-1]

        if hasattr(start, "to_pydatetime"):
            days = (end - start).days
        else:
            days = len(total_series)

        if days <= 0:
            return 0.0
        years = days / 365.25
        if years <= 0:
            return 0.0

        return (math.pow(final / initial, 1 / years) - 1) * 100

    @staticmethod
    def _sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Annualised Sharpe Ratio (assuming 252 trading days)."""
        if len(returns) < 2:
            return 0.0
        excess = returns - risk_free_rate / 252
        std = excess.std()
        if std == 0 or np.isnan(std):
            return 0.0
        return float((excess.mean() / std) * math.sqrt(252))

    @staticmethod
    def _sortino(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Annualised Sortino Ratio (downside deviation only)."""
        if len(returns) < 2:
            return 0.0
        excess = returns - risk_free_rate / 252
        downside = excess[excess < 0]
        if len(downside) < 1:
            return 0.0
        downside_std = downside.std()
        if downside_std == 0 or np.isnan(downside_std):
            return 0.0
        return float((excess.mean() / downside_std) * math.sqrt(252))

    @staticmethod
    def _max_dd_duration(drawdown_series: pd.Series) -> int:
        """Longest consecutive drawdown period in bars."""
        if drawdown_series.empty:
            return 0
        in_dd = drawdown_series < 0
        max_dur = 0
        cur = 0
        for v in in_dd:
            if v:
                cur += 1
                max_dur = max(max_dur, cur)
            else:
                cur = 0
        return max_dur
