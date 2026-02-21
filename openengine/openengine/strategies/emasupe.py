import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


class EmaSupertrendStrategy(BaseStrategy):
    def __init__(
        self,
        fast_ema_period: int = 20,
        slow_ema_period: int = 50,
        atr_period: int = 10,
        atr_multiplier: float = 3.0,
    ):
        self.fast_ema_period = fast_ema_period
        self.slow_ema_period = slow_ema_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        data = data.copy()

        signals = pd.Series(0, index=data.index, dtype=float)

        # EMA calculations
        data["ema_fast"] = data["Close"].ewm(
            span=self.fast_ema_period, adjust=False
        ).mean()
        data["ema_slow"] = data["Close"].ewm(
            span=self.slow_ema_period, adjust=False
        ).mean()

        # True Range and ATR
        data["prev_close"] = data["Close"].shift(1)
        data["tr1"] = data["High"] - data["Low"]
        data["tr2"] = (data["High"] - data["prev_close"]).abs()
        data["tr3"] = (data["Low"] - data["prev_close"]).abs()
        data["tr"] = data[["tr1", "tr2", "tr3"]].max(axis=1)
        data["atr"] = data["tr"].rolling(window=self.atr_period).mean()

        # Supertrend calculation
        hl2 = (data["High"] + data["Low"]) / 2.0
        data["basic_upper"] = hl2 + self.atr_multiplier * data["atr"]
        data["basic_lower"] = hl2 - self.atr_multiplier * data["atr"]

        final_upper = np.zeros(len(data))
        final_lower = np.zeros(len(data))
        supertrend = np.zeros(len(data))

        for i in range(len(data)):
            if i == 0:
                final_upper[i] = data["basic_upper"].iloc[i]
                final_lower[i] = data["basic_lower"].iloc[i]
                supertrend[i] = np.nan
            else:
                final_upper[i] = (
                    data["basic_upper"].iloc[i]
                    if (
                        data["basic_upper"].iloc[i] < final_upper[i - 1]
                        or data["Close"].iloc[i - 1] > final_upper[i - 1]
                    )
                    else final_upper[i - 1]
                )

                final_lower[i] = (
                    data["basic_lower"].iloc[i]
                    if (
                        data["basic_lower"].iloc[i] > final_lower[i - 1]
                        or data["Close"].iloc[i - 1] < final_lower[i - 1]
                    )
                    else final_lower[i - 1]
                )

                if supertrend[i - 1] == final_upper[i - 1]:
                    if data["Close"].iloc[i] <= final_upper[i]:
                        supertrend[i] = final_upper[i]
                    else:
                        supertrend[i] = final_lower[i]
                elif supertrend[i - 1] == final_lower[i - 1]:
                    if data["Close"].iloc[i] >= final_lower[i]:
                        supertrend[i] = final_lower[i]
                    else:
                        supertrend[i] = final_upper[i]
                else:
                    supertrend[i] = np.nan

        data["supertrend"] = supertrend

        # Trend direction from supertrend
        data["st_direction"] = np.where(
            data["Close"] > data["supertrend"], 1, -1
        )

        # Signal conditions
        buy_condition = (
            (data["ema_fast"] > data["ema_slow"])
            & (data["st_direction"] == 1)
        )

        sell_condition = (
            (data["ema_fast"] < data["ema_slow"])
            & (data["st_direction"] == -1)
        )

        signals[buy_condition] = 1.0
        signals[sell_condition] = -1.0

        return signals