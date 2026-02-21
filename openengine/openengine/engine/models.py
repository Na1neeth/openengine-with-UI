"""
Data models for the OpenEngine simulation engine.

All structured types used across the engine modules are defined here:
BacktestConfig, Position, Trade, and BacktestResult.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SizingMode(Enum):
    """Position sizing methods."""
    FIXED_QUANTITY = "fixed_quantity"
    PERCENT_OF_CAPITAL = "percent_of_capital"
    RISK_BASED = "risk_based"


class Direction(Enum):
    """Trade direction."""
    LONG = "long"
    SHORT = "short"


class OrderType(Enum):
    """Supported order types."""
    MARKET = "market"


class ExecutionMode(Enum):
    """When orders execute relative to signal bar."""
    NEXT_BAR_OPEN = "next_bar_open"   # default — prevents lookahead bias
    CURRENT_BAR_CLOSE = "current_bar_close"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Complete configuration for a backtest run."""

    # Capital
    initial_capital: float = 100_000.0

    # Brokerage & slippage
    brokerage_pct: float = 0.0       # e.g. 0.05 = 0.05%  per side
    slippage_pct: float = 0.0        # e.g. 0.01 = 0.01%  per side

    # Position sizing
    sizing_mode: SizingMode = SizingMode.FIXED_QUANTITY
    fixed_quantity: int = 0          # 0 = auto-calculate (max affordable)
    percent_of_capital: float = 100.0  # used when sizing_mode is PERCENT_OF_CAPITAL
    risk_per_trade_pct: float = 2.0   # used when sizing_mode is RISK_BASED

    # Default stop-loss / take-profit (0 = disabled)
    default_sl_pct: float = 0.0      # e.g. 5.0 = 5% stop-loss
    default_tp_pct: float = 0.0      # e.g. 10.0 = 10% take-profit

    # Execution
    execution_mode: ExecutionMode = ExecutionMode.NEXT_BAR_OPEN

    # Direction
    allow_short: bool = False         # whether -1 signals open short positions


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """An open position being tracked by the engine."""

    entry_date: Any = None         # datetime or pd.Timestamp
    entry_price: float = 0.0
    quantity: int = 0
    direction: Direction = Direction.LONG
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # Running state
    unrealised_pnl: float = 0.0

    def market_value(self, current_price: float) -> float:
        """Current market value of this position."""
        if self.direction == Direction.LONG:
            return self.quantity * current_price
        else:
            # Short: profit when price drops
            return self.quantity * (2 * self.entry_price - current_price)

    def unrealised(self, current_price: float) -> float:
        """Unrealised PnL at *current_price*."""
        if self.direction == Direction.LONG:
            return self.quantity * (current_price - self.entry_price)
        else:
            return self.quantity * (self.entry_price - current_price)


# ---------------------------------------------------------------------------
# Completed Trade
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """A completed (closed) trade record."""

    entry_date: Any = None
    exit_date: Any = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: int = 0
    direction: Direction = Direction.LONG

    # PnL
    gross_pnl: float = 0.0
    fees: float = 0.0               # total fees (entry + exit)
    net_pnl: float = 0.0            # gross_pnl - fees

    # Metadata
    holding_period: int = 0          # in bars
    exit_reason: str = ""            # "signal", "stop_loss", "take_profit"

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict for JSON output."""
        fmt = lambda d: d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
        return {
            "entry_date": fmt(self.entry_date),
            "exit_date": fmt(self.exit_date),
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(self.exit_price, 2),
            "quantity": self.quantity,
            "direction": self.direction.value,
            "gross_pnl": round(self.gross_pnl, 2),
            "fees": round(self.fees, 2),
            "net_pnl": round(self.net_pnl, 2),
            "holding_period": self.holding_period,
            "exit_reason": self.exit_reason,
        }


# ---------------------------------------------------------------------------
# Backtest Result
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Structured result returned by Backtester.run()."""

    # Core data
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: List[Trade] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    drawdown_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    # Config snapshot
    config: Optional[BacktestConfig] = None

    def to_dict(self) -> Dict[str, Any]:
        """Full result as a JSON-serialisable dict."""
        eq_dates = []
        eq_values = []
        for idx, row in self.equity_curve.iterrows():
            date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
            eq_dates.append(date_str)
            eq_values.append(round(float(row.get("total", 0)), 2))

        dd_dates = []
        dd_values = []
        for idx, val in self.drawdown_series.items():
            date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
            dd_dates.append(date_str)
            dd_values.append(round(float(val), 4))

        return {
            "equity_curve": {"dates": eq_dates, "values": eq_values},
            "trades": [t.to_dict() for t in self.trades],
            "metrics": self.metrics,
            "drawdown_series": {"dates": dd_dates, "values": dd_values},
        }


# ---------------------------------------------------------------------------
# Out-of-Sample Configuration
# ---------------------------------------------------------------------------

@dataclass
class OOSConfig:
    """Configuration for out-of-sample testing."""

    enabled: bool = False
    split_method: str = "percentage"  # "date" or "percentage"
    split_date: str = ""              # for date-based split (e.g. "2019-01-01")
    train_pct: float = 70.0           # for percentage-based split
    optimize: bool = False            # enable grid-search on training data


# ---------------------------------------------------------------------------
# Out-of-Sample Result
# ---------------------------------------------------------------------------

@dataclass
class OOSResult:
    """Structured result from an out-of-sample backtest."""

    train_result: BacktestResult = field(default_factory=BacktestResult)
    test_result: BacktestResult = field(default_factory=BacktestResult)
    best_params: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    # Metadata
    split_method: str = ""
    split_date: str = ""
    train_pct: float = 0.0
    train_start: str = ""
    train_end: str = ""
    test_start: str = ""
    test_end: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Full OOS result as a JSON-serialisable dict."""
        train_dict = self.train_result.to_dict()
        test_dict = self.test_result.to_dict()

        return {
            "train_metrics": self.train_result.metrics,
            "test_metrics": self.test_result.metrics,
            "train_equity_curve": train_dict["equity_curve"],
            "test_equity_curve": test_dict["equity_curve"],
            "train_trade_log": train_dict["trades"],
            "test_trade_log": test_dict["trades"],
            "train_drawdown": train_dict["drawdown_series"],
            "test_drawdown": test_dict["drawdown_series"],
            "best_params": self.best_params,
            "warnings": self.warnings,
            "split_info": {
                "method": self.split_method,
                "split_date": self.split_date,
                "train_pct": self.train_pct,
                "train_range": f"{self.train_start} to {self.train_end}",
                "test_range": f"{self.test_start} to {self.test_end}",
            },
        }


# ---------------------------------------------------------------------------
# Monte Carlo Configuration
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo robustness testing."""

    enabled: bool = False
    simulations: int = 1000


# ---------------------------------------------------------------------------
# Monte Carlo Result
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloResult:
    """Result from Monte Carlo simulation."""

    final_equity_distribution: List[float] = field(default_factory=list)
    drawdown_distribution: List[float] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    sample_equity_curves: List[List[float]] = field(default_factory=list)
    p5_band: List[float] = field(default_factory=list)
    p95_band: List[float] = field(default_factory=list)
    median_band: List[float] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-friendly dict."""
        return {
            "final_equity_distribution": self.final_equity_distribution,
            "drawdown_distribution": self.drawdown_distribution,
            "statistics": self.statistics,
            "sample_equity_curves": self.sample_equity_curves,
            "p5_band": self.p5_band,
            "p95_band": self.p95_band,
            "median_band": self.median_band,
            "warnings": self.warnings,
        }
