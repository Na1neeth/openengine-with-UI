"""
Position Manager — sizing, stop-loss/take-profit, and position lifecycle.

Responsibilities:
  - Compute position size based on configured sizing mode
  - Create and track open positions
  - Determine stop-loss and take-profit levels
  - Close positions and produce Trade records
"""

from typing import Optional, List

from .models import (
    BacktestConfig,
    Direction,
    Position,
    SizingMode,
    Trade,
)


class PositionManager:
    """Manages position sizing and open-position lifecycle."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.open_position: Optional[Position] = None

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def compute_quantity(
        self,
        price: float,
        available_cash: float,
        direction: Direction = Direction.LONG,
    ) -> int:
        """Return the number of shares to buy/sell based on sizing mode."""

        mode = self.config.sizing_mode

        if mode == SizingMode.FIXED_QUANTITY:
            qty = self.config.fixed_quantity
            if qty <= 0:
                # Auto: max affordable
                qty = int(available_cash // price) if price > 0 else 0
            return qty

        if mode == SizingMode.PERCENT_OF_CAPITAL:
            allocation = available_cash * (self.config.percent_of_capital / 100.0)
            return int(allocation // price) if price > 0 else 0

        if mode == SizingMode.RISK_BASED:
            # Risk-based: size so that hitting the stop-loss loses at most
            # risk_per_trade_pct % of available capital.
            sl_pct = self.config.default_sl_pct
            if sl_pct <= 0:
                # Fallback to max affordable if no SL is set
                return int(available_cash // price) if price > 0 else 0

            max_risk = available_cash * (self.config.risk_per_trade_pct / 100.0)
            risk_per_share = price * (sl_pct / 100.0)
            if risk_per_share <= 0:
                return 0
            return int(max_risk / risk_per_share)

        # Fallback
        return int(available_cash // price) if price > 0 else 0

    # ------------------------------------------------------------------
    # Stop-loss / Take-profit levels
    # ------------------------------------------------------------------

    def compute_sl(self, entry_price: float, direction: Direction) -> Optional[float]:
        """Compute stop-loss price. Returns None if SL is disabled."""
        if self.config.default_sl_pct <= 0:
            return None
        offset = entry_price * (self.config.default_sl_pct / 100.0)
        if direction == Direction.LONG:
            return entry_price - offset
        else:
            return entry_price + offset

    def compute_tp(self, entry_price: float, direction: Direction) -> Optional[float]:
        """Compute take-profit price. Returns None if TP is disabled."""
        if self.config.default_tp_pct <= 0:
            return None
        offset = entry_price * (self.config.default_tp_pct / 100.0)
        if direction == Direction.LONG:
            return entry_price + offset
        else:
            return entry_price - offset

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(
        self,
        date,
        price: float,
        quantity: int,
        direction: Direction = Direction.LONG,
    ) -> Position:
        """Open a new position and track it."""
        pos = Position(
            entry_date=date,
            entry_price=price,
            quantity=quantity,
            direction=direction,
            stop_loss=self.compute_sl(price, direction),
            take_profit=self.compute_tp(price, direction),
        )
        self.open_position = pos
        return pos

    def close(
        self,
        date,
        exit_price: float,
        entry_bar: int,
        current_bar: int,
        entry_fee: float,
        exit_fee: float,
        exit_reason: str = "signal",
    ) -> Optional[Trade]:
        """Close the current open position and return a Trade record."""
        pos = self.open_position
        if pos is None:
            return None

        if pos.direction == Direction.LONG:
            gross = pos.quantity * (exit_price - pos.entry_price)
        else:
            gross = pos.quantity * (pos.entry_price - exit_price)

        total_fees = entry_fee + exit_fee
        net = gross - total_fees

        trade = Trade(
            entry_date=pos.entry_date,
            exit_date=date,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            direction=pos.direction,
            gross_pnl=gross,
            fees=total_fees,
            net_pnl=net,
            holding_period=current_bar - entry_bar,
            exit_reason=exit_reason,
        )
        self.open_position = None
        return trade

    @property
    def has_position(self) -> bool:
        return self.open_position is not None

    def __repr__(self) -> str:
        if self.open_position:
            p = self.open_position
            return (
                f"PositionManager(open={p.direction.value} "
                f"{p.quantity}@{p.entry_price:.2f})"
            )
        return "PositionManager(no position)"
