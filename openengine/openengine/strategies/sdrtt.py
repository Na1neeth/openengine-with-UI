import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy

class BollingerRsiStrategy(BaseStrategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        data = data.copy()

        # Bollinger Bands (20-period, 2 std dev)
        bb_period = 20
        bb_std = 2
        data["bb_mid"] = data["Close"].rolling(window=bb_period).mean()
        data["bb_std"] = data["Close"].rolling(window=bb_period).std()
        data["bb_upper"] = data["bb_mid"] + (bb_std * data["bb_std"])
        data["bb_lower"] = data["bb_mid"] - (bb_std * data["bb_std"])

        # RSI (14-period)
        rsi_period = 14
        delta = data["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        data["rsi"] = 100 - (100 / (1 + rs))

        # Signals
        signals = pd.Series(0, index=data.index, dtype=float)

        # BUY: Price touches lower Bollinger Band AND RSI < 35 (oversold)
        signals.loc[(data["Close"] <= data["bb_lower"]) & (data["rsi"] < 35)] = 1.0

        # SELL: Price touches upper Bollinger Band AND RSI > 65 (overbought)
        signals.loc[(data["Close"] >= data["bb_upper"]) & (data["rsi"] > 65)] = -1.0

        return signals