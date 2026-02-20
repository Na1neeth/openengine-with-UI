import pandas as pd
from openengine.strategies.base_strategy import BaseStrategy

class EmaCrossoverPro(BaseStrategy):
    """EMA Crossover strategy with volume confirmation and configurable spans."""

    def __init__(self, fast_span: int = 9, slow_span: int = 21, vol_window: int = 20):
        self.fast_span = fast_span
        self.slow_span = slow_span
        self.vol_window = vol_window

    def get_parameter_grid(self) -> dict:
        return {
            "fast_span": [5, 9, 12, 15],
            "slow_span": [21, 26, 30, 40],
            "vol_window": [15, 20, 25],
        }

    def generate_signals(self, data):
        signals = pd.Series(0, index=data.index)
        ema_fast = data["Close"].ewm(span=self.fast_span).mean()
        ema_slow = data["Close"].ewm(span=self.slow_span).mean()
        avg_vol = data["Volume"].rolling(self.vol_window).mean()
        for i in range(1, len(data)):
            if (ema_fast.iloc[i] > ema_slow.iloc[i]
                    and ema_fast.iloc[i-1] <= ema_slow.iloc[i-1]
                    and data["Volume"].iloc[i] > avg_vol.iloc[i]):
                signals.iloc[i] = 1
            elif (ema_fast.iloc[i] < ema_slow.iloc[i]
                    and ema_fast.iloc[i-1] >= ema_slow.iloc[i-1]):
                signals.iloc[i] = -1
        return signals