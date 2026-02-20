"""
Execution Engine — candle-by-candle trade simulation with SL/TP support.

Responsibilities:
  - Iterate through OHLCV data bar by bar
  - Execute orders on next-bar open (configurable) to prevent lookahead bias
  - Check stop-loss and take-profit triggers on each bar
  - Apply slippage and brokerage fees
  - Produce Trade records and update CapitalEngine
"""

from typing import List, Tuple

import pandas as pd

from .models import (
    BacktestConfig,
    Direction,
    ExecutionMode,
    Position,
    Trade,
)
from .capital_engine import CapitalEngine
from .position_manager import PositionManager


class ExecutionEngine:
    """Simulates candle-by-candle trade execution."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.capital = CapitalEngine(config.initial_capital)
        self.positions = PositionManager(config)
        self.trades: List[Trade] = []

        # Internal tracking
        self._entry_bar: int = 0      # bar index where current position opened
        self._entry_fee: float = 0.0  # fee paid on entry (stored for trade record)

    # ------------------------------------------------------------------
    # Price helpers (slippage)
    # ------------------------------------------------------------------

    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        """Apply slippage to *price*. Buys get worse (higher), sells get worse (lower)."""
        slip = price * (self.config.slippage_pct / 100.0)
        return price + slip if is_buy else price - slip

    def _compute_fee(self, price: float, quantity: int) -> float:
        """Brokerage fee for a trade leg."""
        return abs(price * quantity * (self.config.brokerage_pct / 100.0))

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
    ) -> Tuple[List[dict], List[Trade]]:
        """Execute the simulation and return (equity_rows, completed_trades).

        Each equity_row is a dict: {date, holdings, cash, total}.
        """
        equity_rows: List[dict] = []
        n = len(data)

        # Determine the signal offset for execution
        # NEXT_BAR_OPEN: signal at bar i executes at bar i+1
        use_next_bar = (self.config.execution_mode == ExecutionMode.NEXT_BAR_OPEN)

        for i in range(n):
            date = data.index[i]
            open_price = self._safe_float(data["Open"].iloc[i])
            high_price = self._safe_float(data["High"].iloc[i])
            low_price = self._safe_float(data["Low"].iloc[i])
            close_price = self._safe_float(data["Close"].iloc[i])

            # -- Determine which signal drives this bar --
            if use_next_bar:
                # Signal from previous bar executes at this bar's open
                signal = float(signals.iloc[i - 1]) if i > 0 else 0.0
                execution_price = open_price
            else:
                signal = float(signals.iloc[i]) if i > 0 else 0.0
                execution_price = close_price

            # -- 1. Check SL / TP on open positions BEFORE processing new signals --
            if self.positions.has_position:
                sl_tp_trade = self._check_sl_tp(
                    date, i, open_price, high_price, low_price
                )
                if sl_tp_trade:
                    self.trades.append(sl_tp_trade)

            # -- 2. Process signal (only if no position currently) --
            if not self.positions.has_position and self.capital.is_tradeable():
                if signal == 1.0:
                    self._open_long(date, i, execution_price)
                elif signal == -1.0 and self.config.allow_short:
                    self._open_short(date, i, execution_price)

            elif self.positions.has_position:
                # Close on opposing signal
                pos = self.positions.open_position
                if pos.direction == Direction.LONG and signal == -1.0:
                    trade = self._close_position(date, i, execution_price, "signal")
                    if trade:
                        self.trades.append(trade)
                elif pos.direction == Direction.SHORT and signal == 1.0:
                    trade = self._close_position(date, i, execution_price, "signal")
                    if trade:
                        self.trades.append(trade)

            # -- 3. Record equity --
            holdings = 0.0
            if self.positions.has_position:
                pos = self.positions.open_position
                holdings = pos.market_value(close_price)

            equity_rows.append({
                "date": date,
                "holdings": holdings,
                "cash": self.capital.available_cash,
                "total": holdings + self.capital.available_cash,
            })

        # -- Force-close any open position at the end --
        if self.positions.has_position:
            last_close = self._safe_float(data["Close"].iloc[-1])
            trade = self._close_position(
                data.index[-1], n - 1, last_close, "end_of_data"
            )
            if trade:
                self.trades.append(trade)
                # Update last equity row
                if equity_rows:
                    equity_rows[-1]["holdings"] = 0.0
                    equity_rows[-1]["cash"] = self.capital.available_cash
                    equity_rows[-1]["total"] = self.capital.available_cash

        return equity_rows, self.trades

    # ------------------------------------------------------------------
    # Internal: open / close positions
    # ------------------------------------------------------------------

    def _open_long(self, date, bar_idx: int, raw_price: float) -> None:
        """Open a LONG position."""
        price = self._apply_slippage(raw_price, is_buy=True)
        qty = self.positions.compute_quantity(
            price, self.capital.available_cash, Direction.LONG
        )
        if qty <= 0:
            return

        cost = price * qty
        fee = self._compute_fee(price, qty)

        # Check we can afford cost + fee
        if cost + fee > self.capital.available_cash:
            qty = int((self.capital.available_cash - fee) // price)
            if qty <= 0:
                return
            cost = price * qty
            fee = self._compute_fee(price, qty)

        self.capital.allocate(cost)
        self.capital.deduct_fee(fee)
        self._entry_fee = fee
        self._entry_bar = bar_idx
        self.positions.open(date, price, qty, Direction.LONG)

    def _open_short(self, date, bar_idx: int, raw_price: float) -> None:
        """Open a SHORT position."""
        price = self._apply_slippage(raw_price, is_buy=False)
        qty = self.positions.compute_quantity(
            price, self.capital.available_cash, Direction.SHORT
        )
        if qty <= 0:
            return

        cost = price * qty  # margin required
        fee = self._compute_fee(price, qty)

        if cost + fee > self.capital.available_cash:
            qty = int((self.capital.available_cash - fee) // price)
            if qty <= 0:
                return
            cost = price * qty
            fee = self._compute_fee(price, qty)

        self.capital.allocate(cost)
        self.capital.deduct_fee(fee)
        self._entry_fee = fee
        self._entry_bar = bar_idx
        self.positions.open(date, price, qty, Direction.SHORT)

    def _close_position(
        self, date, bar_idx: int, raw_price: float, reason: str
    ) -> Trade:
        """Close the current position and return a Trade."""
        pos = self.positions.open_position
        is_buy_to_close = pos.direction == Direction.SHORT
        price = self._apply_slippage(raw_price, is_buy=is_buy_to_close)
        exit_fee = self._compute_fee(price, pos.quantity)

        # Release capital
        original_cost = pos.entry_price * pos.quantity
        if pos.direction == Direction.LONG:
            proceeds = price * pos.quantity - exit_fee
        else:
            # Short: profit = (entry - exit) * qty
            pnl = (pos.entry_price - price) * pos.quantity
            proceeds = original_cost + pnl - exit_fee

        self.capital.release(original_cost, proceeds)

        trade = self.positions.close(
            date=date,
            exit_price=price,
            entry_bar=self._entry_bar,
            current_bar=bar_idx,
            entry_fee=self._entry_fee,
            exit_fee=exit_fee,
            exit_reason=reason,
        )
        self._entry_fee = 0.0
        return trade

    # ------------------------------------------------------------------
    # SL / TP checking
    # ------------------------------------------------------------------

    def _check_sl_tp(
        self, date, bar_idx: int,
        open_price: float, high_price: float, low_price: float,
    ) -> Trade:
        """Check if SL or TP was hit during this bar. Returns Trade or None."""
        pos = self.positions.open_position
        if pos is None:
            return None

        if pos.direction == Direction.LONG:
            # Stop-loss: triggered if low ≤ SL
            if pos.stop_loss is not None and low_price <= pos.stop_loss:
                return self._close_position(date, bar_idx, pos.stop_loss, "stop_loss")
            # Take-profit: triggered if high ≥ TP
            if pos.take_profit is not None and high_price >= pos.take_profit:
                return self._close_position(date, bar_idx, pos.take_profit, "take_profit")
        else:
            # Short SL: triggered if high ≥ SL
            if pos.stop_loss is not None and high_price >= pos.stop_loss:
                return self._close_position(date, bar_idx, pos.stop_loss, "stop_loss")
            # Short TP: triggered if low ≤ TP
            if pos.take_profit is not None and low_price <= pos.take_profit:
                return self._close_position(date, bar_idx, pos.take_profit, "take_profit")

        return None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_float(value) -> float:
        """Convert a value (possibly numpy scalar) to Python float."""
        return float(value.item()) if hasattr(value, "item") else float(value)
