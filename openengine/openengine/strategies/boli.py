import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


class BollingStrategy(BaseStrategy):
    def __init__(self, period: int = 20, std_multiplier: float = 2.0):
        self.period = period
        self.std_multiplier = std_multiplier

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        data = data.copy()

        signals = pd.Series(0, index=data.index, dtype=float)

        # Calculate Bollinger Bands
        rolling_mean = data["Close"].rolling(window=self.period).mean()
        rolling_std = data["Close"].rolling(window=self.period).std()

        upper_band = rolling_mean + (self.std_multiplier * rolling_std)
        lower_band = rolling_mean - (self.std_multiplier * rolling_std)

        # Generate signals
        buy_condition = data["Close"] < lower_band
        sell_condition = data["Close"] > upper_band

        signals[buy_condition] = 1.0
        signals[sell_condition] = -1.0

        return signals