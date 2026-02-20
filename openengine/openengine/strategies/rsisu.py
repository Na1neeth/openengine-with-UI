import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


class RsiStrategy(BaseStrategy):
    def __init__(self, period: int = 14, overbought: float = 70.0, oversold: float = 30.0):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        data = data.copy()
        signals = pd.Series(0.0, index=data.index, dtype=float)

        delta = data["Close"].diff()

        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)

        gain_series = pd.Series(gain, index=data.index)
        loss_series = pd.Series(loss, index=data.index)

        avg_gain = gain_series.rolling(window=self.period, min_periods=self.period).mean()
        avg_loss = loss_series.rolling(window=self.period, min_periods=self.period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        signals[rsi < self.oversold] = 1.0
        signals[rsi > self.overbought] = -1.0

        signals = signals.fillna(0.0)

        return signals
