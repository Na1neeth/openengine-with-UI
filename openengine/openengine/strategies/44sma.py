import pandas as pd
import numpy as np
from openengine.strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def generate_signals(self, data):
        data = data.copy()

        signals = pd.Series(0, index=data.index)

        # 44-period SMA
        sma_44 = data["Close"].rolling(window=44).mean()

        position = 0
        entry_price = 0.0

        for i in range(len(data)):
            if i < 44 or np.isnan(sma_44.iat[i]):
                continue

            low_price = float(data["Low"].iat[i])
            high_price = float(data["High"].iat[i])
            close_price = float(data["Close"].iat[i])
            sma_value = float(sma_44.iat[i])

            if position == 0:
                # Lower part of candle touches 44 SMA
                if low_price <= sma_value <= high_price:
                    signals.iat[i] = 1
                    position = 1
                    entry_price = close_price
            else:
                # Take Profit +6%
                if close_price >= entry_price * 1.06:
                    signals.iat[i] = -1
                    position = 0
                # Stop Loss -3%
                elif close_price <= entry_price * 0.97:
                    signals.iat[i] = -1
                    position = 0

        return signals
