import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


class MeanReversionStrategy2(BaseStrategy):
    def __init__(self, lookback: int = 20, entry_z: float = 2.0):
        self.lookback = lookback
        self.entry_z = entry_z

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        data = data.copy()
        signals = pd.Series(0.0, index=data.index, dtype=float)

        rolling_mean = data["Close"].rolling(window=self.lookback, min_periods=self.lookback).mean()
        rolling_std = data["Close"].rolling(window=self.lookback, min_periods=self.lookback).std()

        z_score = (data["Close"] - rolling_mean) / rolling_std.replace(0, np.nan)

        signals[z_score < -self.entry_z] = 1.0
        signals[z_score > self.entry_z] = -1.0

        signals = signals.fillna(0.0)

        return signals
