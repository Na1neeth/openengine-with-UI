import pandas as pd
import numpy as np
from openengine.strategies.base_strategy import BaseStrategy

class ATRStrategy(BaseStrategy):
    def generate_signals(self, data):
        # Parameters
        atr_period = 14
        ma_period = 50
        atr_multiplier = 1.5

        # Ensure required columns exist
        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        if not required_cols.issubset(data.columns):
            raise ValueError("Input data must contain Open, High, Low, Close, Volume columns")

        df = data.copy()

        # Calculate True Range (TR)
        df["prev_close"] = df["Close"].shift(1)
        df["tr1"] = df["High"] - df["Low"]
        df["tr2"] = (df["High"] - df["prev_close"]).abs()
        df["tr3"] = (df["Low"] - df["prev_close"]).abs()
        df["TR"] = df[["tr1", "tr2", "tr3"]].max(axis=1)

        # Calculate ATR
        df["ATR"] = df["TR"].rolling(window=atr_period, min_periods=atr_period).mean()

        # Trend filter using moving average
        df["MA"] = df["Close"].rolling(window=ma_period, min_periods=ma_period).mean()

        # Volatility breakout logic
        df["upper_band"] = df["MA"] + atr_multiplier * df["ATR"]
        df["lower_band"] = df["MA"] - atr_multiplier * df["ATR"]

        signals = pd.Series(0, index=df.index)

        # Buy when price closes above upper ATR band (volatility expansion in uptrend)
        buy_condition = (df["Close"] > df["upper_band"]) & (df["Close"] > df["MA"])

        # Sell when price closes below lower ATR band (volatility expansion in downtrend)
        sell_condition = (df["Close"] < df["lower_band"]) & (df["Close"] < df["MA"])

        signals[buy_condition] = 1
        signals[sell_condition] = -1

        return signals.fillna(0)
