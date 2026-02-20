"""
Validation Tools — parameter sensitivity, out-of-sample, and walk-forward testing.

These are optional robustness tools that can be run after a standard backtest
to validate strategy reliability.
"""

from typing import List, Dict, Any, Callable, Tuple, Optional
import copy

import pandas as pd

from .models import BacktestConfig, BacktestResult


class ValidationTools:
    """Optional robustness validation methods."""

    @staticmethod
    def parameter_sensitivity(
        data: pd.DataFrame,
        strategy_factory: Callable[..., Any],
        param_name: str,
        param_values: list,
        config: BacktestConfig,
        backtester_class: Any,
    ) -> List[Dict[str, Any]]:
        """Run the same strategy over a range of parameter values.

        Args:
            data: OHLCV DataFrame.
            strategy_factory: Callable that takes **kwargs and returns a strategy.
                             The factory should accept *param_name* as a keyword arg.
            param_name: Name of the parameter to vary.
            param_values: List of values to test.
            config: BacktestConfig to use for all runs.
            backtester_class: The Backtester class.

        Returns:
            List of dicts with keys: param_value, metrics.
        """
        results = []
        for val in param_values:
            try:
                strategy = strategy_factory(**{param_name: val})
                bt = backtester_class(data, strategy, config)
                result = bt.run()
                results.append({
                    "param_value": val,
                    "total_return": result.metrics.get("total_return_pct", 0),
                    "sharpe_ratio": result.metrics.get("sharpe_ratio", 0),
                    "max_drawdown": result.metrics.get("max_drawdown_pct", 0),
                    "win_rate": result.metrics.get("win_rate_pct", 0),
                    "total_trades": result.metrics.get("total_trades", 0),
                })
            except Exception as e:
                results.append({
                    "param_value": val,
                    "error": str(e),
                })
        return results

    @staticmethod
    def out_of_sample(
        data: pd.DataFrame,
        strategy: Any,
        config: BacktestConfig,
        backtester_class: Any,
        split_ratio: float = 0.7,
    ) -> Dict[str, Any]:
        """Split data into in-sample and out-of-sample, run both.

        Args:
            data: full OHLCV DataFrame.
            strategy: strategy instance.
            config: BacktestConfig.
            backtester_class: The Backtester class.
            split_ratio: fraction for in-sample (default 0.7 = 70%).

        Returns:
            Dict with keys: in_sample_metrics, out_of_sample_metrics, split_date.
        """
        n = len(data)
        split_idx = int(n * split_ratio)

        in_sample = data.iloc[:split_idx]
        out_of_sample = data.iloc[split_idx:]

        split_date = str(data.index[split_idx]) if split_idx < n else "N/A"

        # In-sample run
        bt_is = backtester_class(in_sample, strategy, config)
        result_is = bt_is.run()

        # Out-of-sample run
        bt_oos = backtester_class(out_of_sample, strategy, config)
        result_oos = bt_oos.run()

        return {
            "split_date": split_date,
            "in_sample_bars": len(in_sample),
            "out_of_sample_bars": len(out_of_sample),
            "in_sample_metrics": result_is.metrics,
            "out_of_sample_metrics": result_oos.metrics,
        }

    @staticmethod
    def walk_forward(
        data: pd.DataFrame,
        strategy: Any,
        config: BacktestConfig,
        backtester_class: Any,
        n_splits: int = 5,
        in_sample_ratio: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Walk-forward analysis: rolling in-sample / out-of-sample windows.

        Args:
            data: full OHLCV DataFrame.
            strategy: strategy instance.
            config: BacktestConfig.
            backtester_class: The Backtester class.
            n_splits: number of walk-forward windows.
            in_sample_ratio: fraction of each window for in-sample.

        Returns:
            List of dicts, one per window, with in/out-of-sample metrics.
        """
        n = len(data)
        window_size = n // n_splits
        results = []

        for i in range(n_splits):
            start = i * window_size
            end = min(start + window_size, n)
            if end - start < 10:
                continue

            window = data.iloc[start:end]
            split_idx = int(len(window) * in_sample_ratio)

            if split_idx < 5 or len(window) - split_idx < 5:
                continue

            is_data = window.iloc[:split_idx]
            oos_data = window.iloc[split_idx:]

            try:
                bt_is = backtester_class(is_data, strategy, config)
                result_is = bt_is.run()

                bt_oos = backtester_class(oos_data, strategy, config)
                result_oos = bt_oos.run()

                results.append({
                    "window": i + 1,
                    "is_start": str(is_data.index[0]),
                    "is_end": str(is_data.index[-1]),
                    "oos_start": str(oos_data.index[0]),
                    "oos_end": str(oos_data.index[-1]),
                    "in_sample_return": result_is.metrics.get("total_return_pct", 0),
                    "out_of_sample_return": result_oos.metrics.get("total_return_pct", 0),
                    "in_sample_sharpe": result_is.metrics.get("sharpe_ratio", 0),
                    "out_of_sample_sharpe": result_oos.metrics.get("sharpe_ratio", 0),
                })
            except Exception:
                pass

        return results
