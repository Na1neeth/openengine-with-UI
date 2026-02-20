from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on historical data.
        Returns a Pandas Series:
          1 for BUY,
         -1 for SELL,
          0 for HOLD.
        """
        pass

    def get_parameter_grid(self) -> dict:
        """Return dict of param_name -> list of candidate values.

        Override in subclass to enable parameter optimization.
        Returns empty dict by default (no optimization).

        Example::
            return {
                "short_window": [10, 15, 20, 25],
                "long_window":  [40, 50, 60, 80],
            }
        """
        return {}

    def set_params(self, **params) -> None:
        """Apply parameter values to this strategy instance.

        Override in subclass if custom logic is needed.
        Default implementation sets each key as an attribute.
        """
        for key, value in params.items():
            setattr(self, key, value)

    def generate_signal_from_data_point(self, data_point: dict) -> int:
        # Dummy implementation for live trading; override this in concrete strategies.
        return 0

