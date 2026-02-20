"""Default configuration values for OpenEngine."""

INITIAL_CAPITAL = 100000
DEFAULT_INTERVAL = "1d"

# Brokerage & slippage defaults (in %)
DEFAULT_BROKERAGE_PCT = 0.05     # 0.05% per side
DEFAULT_SLIPPAGE_PCT = 0.01      # 0.01% per side

# Position sizing defaults
DEFAULT_SIZING_MODE = "fixed_quantity"  # fixed_quantity | percent_of_capital | risk_based
DEFAULT_FIXED_QUANTITY = 0               # 0 = auto (max affordable)
DEFAULT_PERCENT_OF_CAPITAL = 100.0       # 100% = use all available
DEFAULT_RISK_PER_TRADE_PCT = 2.0         # for risk_based sizing

# Stop-loss / Take-profit defaults (0 = disabled)
DEFAULT_SL_PCT = 0.0
DEFAULT_TP_PCT = 0.0
