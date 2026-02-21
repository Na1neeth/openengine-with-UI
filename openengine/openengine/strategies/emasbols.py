import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


class RsiBollingerStrategy(BaseStrategy):
    def __init__(
        self,
        rsi_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
    ):
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        data = data.copy()

        signals = pd.Series(0, index=data.index, dtype=float)

        # RSI calculation
        delta = data["Close"].diff()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)

        gain_rolling = pd.Series(gain, index=data.index).rolling(
            window=self.rsi_period
        ).mean()
        loss_rolling = pd.Series(loss, index=data.index).rolling(
            window=self.rsi_period
        ).mean()

        rs = gain_rolling / loss_rolling
        data["rsi"] = 100 - (100 / (1 + rs))

        # Bollinger Bands calculation
        data["bb_mid"] = data["Close"].rolling(window=self.bb_period).mean()
        data["bb_std"] = data["Close"].rolling(window=self.bb_period).std()
        data["bb_upper"] = data["bb_mid"] + self.bb_std * data["bb_std"]
        data["bb_lower"] = data["bb_mid"] - self.bb_std * data["bb_std"]

        # Buy: price below lower band AND RSI oversold
        buy_condition = (
            (data["Close"] < data["bb_lower"])
            & (data["rsi"] < self.rsi_oversold)
        )

        # Sell: price above upper band AND RSI overbought
        sell_condition = (
            (data["Close"] > data["bb_upper"])
            & (data["rsi"] > self.rsi_overbought)
        )

        signals[buy_condition] = 1.0
        signals[sell_condition] = -1.0

        return signals