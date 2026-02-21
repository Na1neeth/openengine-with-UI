import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


class EmaCrossoverRsiStrategy(BaseStrategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        data = data.copy()

        signals = pd.Series(0, index=data.index, dtype=float)

        # Parameters (in number of candles)
        short_ema_period = 20
        long_ema_period = 50
        rsi_period = 14
        rsi_overbought = 70
        rsi_oversold = 30

        # EMA calculations
        data["ema_short"] = data["Close"].ewm(span=short_ema_period, adjust=False).mean()
        data["ema_long"] = data["Close"].ewm(span=long_ema_period, adjust=False).mean()

        # RSI calculation
        delta = data["Close"].diff()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)

        gain_series = pd.Series(gain, index=data.index)
        loss_series = pd.Series(loss, index=data.index)

        avg_gain = gain_series.rolling(window=rsi_period, min_periods=rsi_period).mean()
        avg_loss = loss_series.rolling(window=rsi_period, min_periods=rsi_period).mean()

        rs = avg_gain / avg_loss
        data["rsi"] = 100 - (100 / (1 + rs))

        # Crossover conditions
        ema_bullish_cross = (
            (data["ema_short"] > data["ema_long"]) &
            (data["ema_short"].shift(1) <= data["ema_long"].shift(1))
        )

        ema_bearish_cross = (
            (data["ema_short"] < data["ema_long"]) &
            (data["ema_short"].shift(1) >= data["ema_long"].shift(1))
        )

        # Signal logic with RSI confirmation
        buy_condition = ema_bullish_cross & (data["rsi"] < rsi_overbought)
        sell_condition = ema_bearish_cross & (data["rsi"] > rsi_oversold)

        signals[buy_condition] = 1.0
        signals[sell_condition] = -1.0

        return signals