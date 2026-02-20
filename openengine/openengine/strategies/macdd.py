import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


class MacdSt(BaseStrategy):
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        data = data.copy()

        signals = pd.Series(0, index=data.index, dtype=float)

        close = data["Close"]

        ema_fast = close.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_period, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()

        macd_prev = macd_line.shift(1)
        signal_prev = signal_line.shift(1)

        bullish_crossover = (macd_line > signal_line) & (macd_prev <= signal_prev)
        bearish_crossover = (macd_line < signal_line) & (macd_prev >= signal_prev)

        signals[bullish_crossover] = 1.0
        signals[bearish_crossover] = -1.0

        return signals
