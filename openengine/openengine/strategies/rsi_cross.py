import pandas as pd
import numpy as np
from openengine.strategies.base_strategy import BaseStrategy


class RSICrossoverStrategy(BaseStrategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        data = data.copy()
        
        period = 14
        oversold = 30
        overbought = 70

        # RSI calculation (Wilder's smoothing via EMA)
        delta = data["Close"].diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        data["rsi"] = 100 - (100 / (1 + rs))

        signals = pd.Series(0.0, index=data.index)

        # Crossover logic
        prev_rsi = data["rsi"].shift(1)

        # BUY: RSI crosses upward through oversold
        buy_condition = (prev_rsi < oversold) & (data["rsi"] >= oversold)

        # SELL: RSI crosses downward through overbought
        sell_condition = (prev_rsi > overbought) & (data["rsi"] <= overbought)

        signals.loc[buy_condition] = 1.0
        signals.loc[sell_condition] = -1.0

        # Ensure warm-up period returns 0
        signals[data["rsi"].isna()] = 0.0

        return signals
