import pandas as pd
import numpy as np
from openengine.strategies.base_strategy import BaseStrategy

class BuyOnDipStrategy(BaseStrategy):
    def generate_signals(self, data):
        # Parameters
        ma_period = 50
        dip_threshold = 0.03      # 3% below moving average
        rsi_period = 14
        oversold_level = 30
        overbought_level = 70

        # Validate input
        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        if not required_cols.issubset(data.columns):
            raise ValueError("Input data must contain Open, High, Low, Close, Volume columns")

        df = data.copy()

        # Moving Average (trend filter)
        df["MA"] = df["Close"].rolling(window=ma_period, min_periods=ma_period).mean()

        # RSI calculation
        delta = df["Close"].diff()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)

        avg_gain = pd.Series(gain, index=df.index).rolling(rsi_period, min_periods=rsi_period).mean()
        avg_loss = pd.Series(loss, index=df.index).rolling(rsi_period, min_periods=rsi_period).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        df["RSI"] = 100 - (100 / (1 + rs))

        # Dip condition: price below MA by threshold and RSI oversold
        dip_condition = (
            (df["Close"] < df["MA"] * (1 - dip_threshold)) &
            (df["RSI"] < oversold_level)
        )

        # Exit condition: price recovers above MA or RSI overbought
        exit_condition = (
            (df["Close"] > df["MA"]) |
            (df["RSI"] > overbought_level)
        )

        signals = pd.Series(0, index=df.index)
        signals[dip_condition] = 1
        signals[exit_condition] = -1

        return signals.fillna(0)
