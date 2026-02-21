import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


class RsiEmaCrosserStrategy(BaseStrategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        data = data.copy()

        signals = pd.Series(0, index=data.index, dtype=float)

        # Parameters (in number of candles)
        rsi_period = 14
        ema_short_period = 20
        ema_long_period = 50

        # Calculate EMA
        data["ema_short"] = data["Close"].ewm(span=ema_short_period, adjust=False).mean()
        data["ema_long"] = data["Close"].ewm(span=ema_long_period, adjust=False).mean()

        # Calculate RSI
        delta = data["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        data["rsi"] = 100 - (100 / (1 + rs))

        # Generate signals
        ema_cross_up = (data["ema_short"] > data["ema_long"]) & (
            data["ema_short"].shift(1) <= data["ema_long"].shift(1)
        )
        ema_cross_down = (data["ema_short"] < data["ema_long"]) & (
            data["ema_short"].shift(1) >= data["ema_long"].shift(1)
        )

        buy_condition = ema_cross_up & (data["rsi"] > 50)
        sell_condition = ema_cross_down & (data["rsi"] < 50)

        signals[buy_condition] = 1.0
        signals[sell_condition] = -1.0

        return signals