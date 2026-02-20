"""
Optimizer — grid-search parameter optimization on training data.

Uses the existing Backtester engine to evaluate each parameter combination.
Only training data is ever passed to this module.
"""

import copy
import itertools
from typing import Tuple, Dict, Any

import pandas as pd

from .backtester import Backtester
from .models import BacktestConfig, BacktestResult


class Optimizer:
    """Grid-search optimizer for strategy parameters."""

    @staticmethod
    def grid_search(
        train_data: pd.DataFrame,
        strategy,
        config: BacktestConfig,
        metric: str = "sharpe_ratio",
    ) -> Tuple[Dict[str, Any], BacktestResult]:
        """Run grid search over strategy parameter space on training data.

        Args:
            train_data: OHLCV training DataFrame (MUST NOT include test data).
            strategy: Strategy instance with get_parameter_grid() method.
            config: BacktestConfig for running each trial.
            metric: Metric name to optimise (from MetricsEngine output).
                    Defaults to "sharpe_ratio". Other options:
                    "total_return_pct", "profit_factor", "sortino_ratio", etc.

        Returns:
            (best_params, best_result) where best_params is the dict of
            parameter values that produced the highest metric value.
        """
        param_grid = strategy.get_parameter_grid()

        # If no parameter grid defined, run single backtest with current params
        if not param_grid:
            backtester = Backtester(train_data, strategy, config)
            result = backtester.run()
            return {}, result

        # Build all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        best_metric_value = float("-inf")
        best_params: Dict[str, Any] = {}
        best_result: BacktestResult = None

        for combo in combinations:
            params = dict(zip(param_names, combo))

            # Apply parameters to strategy
            strategy.set_params(**params)

            # Run backtest on training data with a fresh engine
            backtester = Backtester(train_data, strategy, config)
            result = backtester.run()

            # Extract the target metric
            metric_value = result.metrics.get(metric, float("-inf"))

            # Handle edge cases (NaN, inf)
            if metric_value is None or metric_value != metric_value:  # NaN check
                metric_value = float("-inf")

            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_params = params.copy()
                best_result = result

        # If no valid result found, run with first combination
        if best_result is None:
            first_combo = dict(zip(param_names, combinations[0]))
            strategy.set_params(**first_combo)
            backtester = Backtester(train_data, strategy, config)
            best_result = backtester.run()
            best_params = first_combo

        return best_params, best_result
