"""
Equity Tracker — builds equity curve and drawdown series.

Responsibilities:
  - Convert raw equity rows into a DataFrame
  - Track running peak equity
  - Compute drawdown percentage at each bar
  - Report max drawdown and drawdown duration
"""

from typing import List, Dict, Any, Tuple

import pandas as pd


class EquityTracker:
    """Builds the equity curve and computes drawdown metrics."""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital

    def build(self, equity_rows: List[dict]) -> Tuple[pd.DataFrame, pd.Series]:
        """Build equity curve DataFrame and drawdown Series.

        Args:
            equity_rows: list of dicts with keys {date, holdings, cash, total}

        Returns:
            (equity_df, drawdown_series)
        """
        if not equity_rows:
            return pd.DataFrame(), pd.Series(dtype=float)

        df = pd.DataFrame(equity_rows)
        df.set_index("date", inplace=True)

        # Compute drawdown
        total = df["total"]
        running_peak = total.cummax()
        drawdown = (total - running_peak) / running_peak  # negative values

        return df, drawdown

    @staticmethod
    def max_drawdown(drawdown_series: pd.Series) -> float:
        """Return the worst (most negative) drawdown as a positive percentage.

        E.g. returns 15.3 for a -15.3% drawdown.
        """
        if drawdown_series.empty:
            return 0.0
        return abs(float(drawdown_series.min())) * 100

    @staticmethod
    def max_drawdown_duration(drawdown_series: pd.Series) -> int:
        """Return the longest drawdown duration in bars.

        A drawdown period starts when drawdown < 0 and ends when it returns to 0.
        """
        if drawdown_series.empty:
            return 0

        in_drawdown = drawdown_series < 0
        max_dur = 0
        current_dur = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_dur += 1
                max_dur = max(max_dur, current_dur)
            else:
                current_dur = 0

        return max_dur
