"""
Monte Carlo Engine — robustness testing via trade-return shuffling.

Operates purely on finished trade results. Does NOT modify the backtest
engine or introduce any lookahead bias. All heavy computation is vectorised
with NumPy for sub-second performance at 1 000+ simulations.

Usage:
    from openengine.engine.monte_carlo_engine import get_trade_returns, run_monte_carlo

    returns = get_trade_returns(result.trades)
    mc = run_monte_carlo(returns, initial_capital=100_000, simulations=1000)
    print(mc.statistics)
"""

from typing import List

import numpy as np

from .models import Trade, Direction, MonteCarloResult


# ---------------------------------------------------------------------------
# 1. Extract trade returns
# ---------------------------------------------------------------------------

def get_trade_returns(trades: List[Trade]) -> List[float]:
    """Extract percentage return per completed trade.

    Uses net PnL (after fees) relative to entry cost:
        return = net_pnl / (entry_price * quantity)

    Args:
        trades: list of completed Trade objects from a backtest.

    Returns:
        List of per-trade fractional returns (e.g. 0.02 = +2%).
    """
    returns = []
    for t in trades:
        cost = t.entry_price * t.quantity
        if cost > 0:
            returns.append(t.net_pnl / cost)
    return returns


# ---------------------------------------------------------------------------
# 2. Monte Carlo simulation (fully vectorised)
# ---------------------------------------------------------------------------

def run_monte_carlo(
    trade_returns: List[float],
    initial_capital: float,
    simulations: int = 1000,
) -> MonteCarloResult:
    """Run Monte Carlo simulation by shuffling trade returns.

    For each simulation:
      1. Randomly permute the trade_returns sequence
      2. Rebuild equity curve: equity *= (1 + return_i) for each trade
      3. Record final equity and maximum drawdown

    All simulations are vectorised — no Python loop over simulations.

    Args:
        trade_returns: list of per-trade fractional returns.
        initial_capital: starting equity.
        simulations: number of Monte Carlo iterations (default 1000).

    Returns:
        MonteCarloResult with distributions and statistics.
    """
    n_trades = len(trade_returns)

    # Edge case: no trades
    if n_trades == 0:
        return MonteCarloResult(
            final_equity_distribution=[initial_capital],
            drawdown_distribution=[0.0],
            statistics=_empty_stats(initial_capital),
            sample_equity_curves=[],
            warnings=["No trades to simulate."],
        )

    returns = np.array(trade_returns, dtype=np.float64)

    # Build shuffled matrix: (simulations × n_trades)
    # Each row is a random permutation of the returns
    idx = np.argsort(np.random.rand(simulations, n_trades), axis=1)
    shuffled = returns[idx]  # advanced indexing

    # Equity curve: cumulative product of (1 + r)
    growth = 1.0 + shuffled                            # (S, T)
    cum_equity = initial_capital * np.cumprod(growth, axis=1)  # (S, T)

    # Final equity for each simulation
    final_equities = cum_equity[:, -1]                 # (S,)

    # Max drawdown per simulation
    running_peak = np.maximum.accumulate(cum_equity, axis=1)
    drawdowns = (cum_equity - running_peak) / running_peak  # negative fractions
    max_drawdowns = np.abs(drawdowns.min(axis=1)) * 100     # as positive %

    # ── Statistics ────────────────────────────────────────────────────
    final_returns_pct = ((final_equities - initial_capital) / initial_capital) * 100

    stats = {
        "median_return": round(float(np.median(final_returns_pct)), 2),
        "mean_return": round(float(np.mean(final_returns_pct)), 2),
        "p5_return": round(float(np.percentile(final_returns_pct, 5)), 2),
        "p95_return": round(float(np.percentile(final_returns_pct, 95)), 2),
        "worst_return": round(float(np.min(final_returns_pct)), 2),
        "best_return": round(float(np.max(final_returns_pct)), 2),
        "probability_negative": round(float(np.mean(final_returns_pct < 0) * 100), 2),
        "median_max_drawdown": round(float(np.median(max_drawdowns)), 2),
        "p95_max_drawdown": round(float(np.percentile(max_drawdowns, 95)), 2),
        "worst_max_drawdown": round(float(np.max(max_drawdowns)), 2),
        "simulations": simulations,
        "n_trades": n_trades,
    }

    # ── Sample equity curves (50 random for plotting) ─────────────────
    n_samples = min(50, simulations)
    sample_idx = np.random.choice(simulations, size=n_samples, replace=False)

    # Prepend initial capital so curves start at the same point
    init_col = np.full((simulations, 1), initial_capital)
    full_equity = np.hstack([init_col, cum_equity])     # (S, T+1)

    sample_curves = full_equity[sample_idx].tolist()

    # ── Percentile bands (5th and 95th) ───────────────────────────────
    p5_band = np.percentile(full_equity, 5, axis=0).tolist()
    p95_band = np.percentile(full_equity, 95, axis=0).tolist()
    median_band = np.median(full_equity, axis=0).tolist()

    # ── Fragility warnings ────────────────────────────────────────────
    warnings = _generate_warnings(stats)

    return MonteCarloResult(
        final_equity_distribution=final_equities.tolist(),
        drawdown_distribution=max_drawdowns.tolist(),
        statistics=stats,
        sample_equity_curves=sample_curves,
        p5_band=p5_band,
        p95_band=p95_band,
        median_band=median_band,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _empty_stats(initial_capital: float) -> dict:
    """Return zeroed-out stats dict for edge cases."""
    return {
        "median_return": 0.0,
        "mean_return": 0.0,
        "p5_return": 0.0,
        "p95_return": 0.0,
        "worst_return": 0.0,
        "best_return": 0.0,
        "probability_negative": 0.0,
        "median_max_drawdown": 0.0,
        "p95_max_drawdown": 0.0,
        "worst_max_drawdown": 0.0,
        "simulations": 0,
        "n_trades": 0,
    }


def _generate_warnings(stats: dict) -> List[str]:
    """Generate fragility alerts based on Monte Carlo statistics."""
    warnings = []

    if stats["p5_return"] < 0:
        warnings.append(
            f"⚠️ 5th percentile return is negative ({stats['p5_return']:.1f}%). "
            f"There is a meaningful risk of losing money with this strategy."
        )

    if stats["p95_max_drawdown"] > 50:
        warnings.append(
            f"🔴 95th percentile max drawdown is {stats['p95_max_drawdown']:.1f}%. "
            f"Extreme drawdowns are likely under adverse trade sequencing."
        )

    if stats["probability_negative"] > 30:
        warnings.append(
            f"⚠️ {stats['probability_negative']:.0f}% probability of negative return. "
            f"Strategy is fragile to trade ordering."
        )

    if stats["worst_return"] < -50:
        warnings.append(
            f"🔴 Worst-case return is {stats['worst_return']:.1f}%. "
            f"Catastrophic loss possible under worst-case sequencing."
        )

    return warnings
