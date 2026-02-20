import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy

class SampleStrategy(BaseStrategy):
    """SMA Crossover strategy with configurable window parameters."""

    def __init__(self, short_window: int = 20, long_window: int = 50):
        self.short_window = short_window
        self.long_window = long_window

    def get_parameter_grid(self) -> dict:
        return {
            "short_window": [10, 15, 20, 25, 30],
            "long_window": [40, 50, 60, 80],
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        data = data.copy()
        data["short_ma"] = data["Close"].rolling(window=self.short_window).mean()
        data["long_ma"] = data["Close"].rolling(window=self.long_window).mean()

        signals = pd.Series(0, index=data.index, dtype=float)
        signals.loc[data["short_ma"] > data["long_ma"]] = 1.0
        signals.loc[data["short_ma"] < data["long_ma"]] = -1.0

        return signals

    def generate_signal_from_data_point(self, data_point: dict) -> float:
        # A simple dummy signal for live data:
        price = float(data_point.get("price", 0))
        return 1.0 if price % 2 == 0 else -1.0
