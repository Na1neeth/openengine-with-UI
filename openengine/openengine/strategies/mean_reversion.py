import pandas as pd
import numpy as np
from openengine.strategies.base_strategy import BaseStrategy

class MeanReversionStrategy(BaseStrategy):
    def generate_signals(self, data):
        # Parameters
        window = 20
        z_threshold = 2.0

        # Initialize signals
        signals = pd.Series(0, index=data.index)

        # Calculate rolling mean and standard deviation
        rolling_mean = data["Close"].rolling(window=window, min_periods=window).mean()
        rolling_std = data["Close"].rolling(window=window, min_periods=window).std()

        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)

        # Calculate z-score
        z_score = (data["Close"] - rolling_mean) / rolling_std

        # Generate mean reversion signals
        signals[z_score > z_threshold] = -1   # SELL when price is significantly above mean
        signals[z_score < -z_threshold] = 1   # BUY when price is significantly below mean

        return signals.fillna(0)
