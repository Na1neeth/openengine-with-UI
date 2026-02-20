"""
Capital Engine — tracks cash flow and prevents over-leveraging.

Responsibilities:
  - Track initial, available, and used capital
  - Allocate capital for new positions
  - Release capital when positions close
  - Stop trading when capital ≤ 0
"""


class CapitalEngine:
    """Manages capital allocation throughout a backtest."""

    def __init__(self, initial_capital: float):
        self.initial_capital = float(initial_capital)
        self.available_cash = float(initial_capital)
        self.used_capital = 0.0

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_tradeable(self) -> bool:
        """Return False if there is no cash left to trade."""
        return self.available_cash > 0

    @property
    def total_capital(self) -> float:
        """Available + used (does not include unrealised PnL)."""
        return self.available_cash + self.used_capital

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def allocate(self, amount: float) -> None:
        """Reserve *amount* of cash for a new position.

        Raises ValueError if insufficient cash.
        """
        if amount <= 0:
            return
        if amount > self.available_cash:
            raise ValueError(
                f"Insufficient capital: need {amount:.2f}, "
                f"available {self.available_cash:.2f}"
            )
        self.available_cash -= amount
        self.used_capital += amount

    def release(self, original_cost: float, proceeds: float) -> None:
        """Return capital when a position is closed.

        *original_cost*: The capital originally allocated (entry cost).
        *proceeds*:      The actual cash received on exit (after fees).
        """
        self.used_capital -= original_cost
        self.available_cash += proceeds

    def deduct_fee(self, fee: float) -> None:
        """Deduct a fee from available cash (e.g. brokerage)."""
        self.available_cash -= fee

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def max_affordable_quantity(self, price: float) -> int:
        """Maximum whole shares affordable at *price*."""
        if price <= 0:
            return 0
        return int(self.available_cash // price)

    def __repr__(self) -> str:
        return (
            f"CapitalEngine(available={self.available_cash:.2f}, "
            f"used={self.used_capital:.2f})"
        )
