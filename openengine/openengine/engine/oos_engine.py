"""
Out-of-Sample Engine — orchestrates train/test backtesting with strict separation.

This is a thin wrapper around the existing Backtester. It does NOT duplicate
any engine logic. The workflow is:

  1. Split data chronologically
  2. Optimise parameters on training data (optional)
  3. Freeze parameters
  4. Run backtest on test data with a fresh engine
  5. Compare metrics and generate degradation warnings
  6. Return structured OOSResult
"""

from typing import Optional

import pandas as pd

from .backtester import Backtester
from .data_splitter import split_data
from .optimizer import Optimizer
from .models import BacktestConfig, OOSConfig, OOSResult


def run_out_of_sample_backtest(
    data: pd.DataFrame,
    strategy,
    config: BacktestConfig,
    oos_config: OOSConfig,
) -> OOSResult:
    """Run a full out-of-sample backtest.

    Args:
        data: Full OHLCV DataFrame.
        strategy: Strategy instance (BaseStrategy subclass).
        config: BacktestConfig for the simulation.
        oos_config: OOS settings (split method, optimisation flag, etc.).

    Returns:
        OOSResult with train/test metrics, equity curves, trade logs,
        best parameters, and degradation warnings.
    """
    # ── 1. Split data ─────────────────────────────────────────────────
    train_data, test_data = split_data(
        data=data,
        split_method=oos_config.split_method,
        split_date=oos_config.split_date if oos_config.split_method == "date" else None,
        train_pct=oos_config.train_pct,
    )

    # ── 2. Training phase ─────────────────────────────────────────────
    if oos_config.optimize:
        best_params, train_result = Optimizer.grid_search(
            train_data=train_data,
            strategy=strategy,
            config=config,
            metric="sharpe_ratio",
        )
    else:
        # Run single backtest with current strategy params
        backtester = Backtester(train_data, strategy, config)
        train_result = backtester.run()
        best_params = {}

    # ── 3. Freeze parameters ──────────────────────────────────────────
    # Apply best params (if optimisation was done) and make no further changes
    if best_params:
        strategy.set_params(**best_params)

    # ── 4. Testing phase (fresh engine, completely isolated) ───────────
    test_backtester = Backtester(test_data, strategy, config)
    test_result = test_backtester.run()

    # ── 5. Degradation warnings ───────────────────────────────────────
    warnings = _check_degradation(train_result.metrics, test_result.metrics)

    # ── 6. Build date range metadata ──────────────────────────────────
    fmt = lambda idx: idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)

    oos_result = OOSResult(
        train_result=train_result,
        test_result=test_result,
        best_params=best_params,
        warnings=warnings,
        split_method=oos_config.split_method,
        split_date=oos_config.split_date,
        train_pct=oos_config.train_pct,
        train_start=fmt(train_data.index[0]),
        train_end=fmt(train_data.index[-1]),
        test_start=fmt(test_data.index[0]),
        test_end=fmt(test_data.index[-1]),
    )

    return oos_result


# ──────────────────────────────────────────────────────────────────────────
# Degradation detection
# ──────────────────────────────────────────────────────────────────────────

def _check_degradation(train_metrics: dict, test_metrics: dict) -> list:
    """Compare train and test metrics and generate warnings for significant drops.

    Thresholds:
    - Total return drop > 50%
    - Sharpe ratio drops > 50% or goes negative when train was positive
    - Max drawdown increases > 50% (relative)
    """
    warnings = []

    # --- Return degradation ---
    train_return = train_metrics.get("total_return_pct", 0)
    test_return = test_metrics.get("total_return_pct", 0)

    if train_return > 0 and test_return < train_return * 0.5:
        drop_pct = ((train_return - test_return) / abs(train_return)) * 100
        warnings.append(
            f"⚠️ Return degradation: test return ({test_return:.1f}%) dropped "
            f"{drop_pct:.0f}% vs train ({train_return:.1f}%)"
        )

    if train_return > 0 and test_return < 0:
        warnings.append(
            f"🔴 Test period is NEGATIVE ({test_return:.1f}%) while "
            f"train was positive ({train_return:.1f}%)"
        )

    # --- Sharpe degradation ---
    train_sharpe = train_metrics.get("sharpe_ratio", 0)
    test_sharpe = test_metrics.get("sharpe_ratio", 0)

    if train_sharpe > 0:
        if test_sharpe < 0:
            warnings.append(
                f"🔴 Sharpe ratio went negative: test ({test_sharpe:.2f}) "
                f"vs train ({train_sharpe:.2f})"
            )
        elif test_sharpe < train_sharpe * 0.5:
            sharpe_drop = ((train_sharpe - test_sharpe) / train_sharpe) * 100
            warnings.append(
                f"⚠️ Sharpe degradation: test ({test_sharpe:.2f}) dropped "
                f"{sharpe_drop:.0f}% vs train ({train_sharpe:.2f})"
            )

    # --- Drawdown degradation ---
    train_dd = train_metrics.get("max_drawdown_pct", 0)
    test_dd = test_metrics.get("max_drawdown_pct", 0)

    if train_dd > 0 and test_dd > train_dd * 1.5:
        dd_increase = ((test_dd - train_dd) / train_dd) * 100
        warnings.append(
            f"⚠️ Drawdown increased: test ({test_dd:.1f}%) is "
            f"{dd_increase:.0f}% worse than train ({train_dd:.1f}%)"
        )
    elif train_dd == 0 and test_dd > 10:
        warnings.append(
            f"⚠️ New drawdown in test period: {test_dd:.1f}% "
            f"(train had no drawdown)"
        )

    return warnings
