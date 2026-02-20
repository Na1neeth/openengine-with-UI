"""
Backtester — thin orchestrator that composes all engine modules.

Usage:
    from openengine.engine.backtester import Backtester
    from openengine.engine.models import BacktestConfig

    config = BacktestConfig(initial_capital=100000, brokerage_pct=0.05)
    bt = Backtester(data, strategy, config)
    result = bt.run()   # → BacktestResult

    print(result.metrics)
    print(result.trades)
"""

import pandas as pd

from .models import BacktestConfig, BacktestResult
from .execution_engine import ExecutionEngine
from .equity_tracker import EquityTracker
from .metrics_engine import MetricsEngine


class Backtester:
    """Professional-grade backtesting engine.

    Orchestrates: signal generation → execution → equity tracking → metrics.
    """

    def __init__(self, data: pd.DataFrame, strategy, config=None, initial_capital: float = None):
        """Initialise the backtester.

        Args:
            data: OHLCV DataFrame (columns: Open, High, Low, Close, Volume).
            strategy: Strategy instance with ``generate_signals(data)`` method.
            config: BacktestConfig (preferred). If None, a default is created.
            initial_capital: Legacy parameter — used if config is None.
        """
        self.data = data
        self.strategy = strategy

        # Backward compatibility: accept old (data, strategy, initial_capital) signature
        if config is None:
            cap = initial_capital if initial_capital is not None else 100_000.0
            self.config = BacktestConfig(initial_capital=cap)
        elif isinstance(config, (int, float)):
            # Called as Backtester(data, strategy, 200000) — treat as capital
            self.config = BacktestConfig(initial_capital=float(config))
        else:
            self.config = config

        self.result: BacktestResult = None

    def run(self) -> BacktestResult:
        """Run the full backtest and return a BacktestResult."""

        # 1. Generate signals from strategy
        signals = self.strategy.generate_signals(self.data)

        # 2. Execute trades candle-by-candle
        exec_engine = ExecutionEngine(self.config)
        equity_rows, trades = exec_engine.run(self.data, signals)

        # 3. Build equity curve and drawdown
        tracker = EquityTracker(self.config.initial_capital)
        equity_curve, drawdown_series = tracker.build(equity_rows)

        # 4. Compute metrics
        metrics = MetricsEngine.compute(
            equity_curve=equity_curve,
            trades=trades,
            drawdown_series=drawdown_series,
            initial_capital=self.config.initial_capital,
        )

        # 5. Package result
        self.result = BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            metrics=metrics,
            drawdown_series=drawdown_series,
            config=self.config,
        )
        return self.result
