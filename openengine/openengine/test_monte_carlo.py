import unittest
from datetime import datetime
import numpy as np

from openengine.engine.models import Trade, Direction
from openengine.engine.monte_carlo_engine import get_trade_returns, run_monte_carlo


class TestMonteCarloEngine(unittest.TestCase):
    def setUp(self):
        # Create some dummy trades
        # Trade 1: Long, entry 100, exit 110, qty 10. Cost = 1000. Net PnL = 100. Return = 10%
        t1 = Trade(
            entry_date=datetime(2023, 1, 1),
            exit_date=datetime(2023, 1, 2),
            entry_price=100.0,
            exit_price=110.0,
            quantity=10.0,
            direction=Direction.LONG,
            gross_pnl=100.0,
            fees=0.0,
            net_pnl=100.0,
            holding_period=1,
            exit_reason="tp"
        )
        # Trade 2: Short, entry 100, exit 90, qty 10. Cost = 1000. Net PnL = 100. Return = 10%
        t2 = Trade(
            entry_date=datetime(2023, 1, 3),
            exit_date=datetime(2023, 1, 4),
            entry_price=100.0,
            exit_price=90.0,
            quantity=10.0,
            direction=Direction.SHORT,
            gross_pnl=100.0,
            fees=0.0,
            net_pnl=100.0,
            holding_period=1,
            exit_reason="sl"
        )
        # Trade 3: Long, entry 200, exit 180, qty 5. Cost = 1000. Net PnL = -100. Return = -10%. (With fees -10) Net = -110 => -11%
        t3 = Trade(
            entry_date=datetime(2023, 1, 5),
            exit_date=datetime(2023, 1, 6),
            entry_price=200.0,
            exit_price=180.0,
            quantity=5.0,
            direction=Direction.LONG,
            gross_pnl=-100.0,
            fees=10.0,
            net_pnl=-110.0,
            holding_period=1,
            exit_reason="sl"
        )
        self.trades = [t1, t2, t3]

    def test_get_trade_returns(self):
        """Test extraction of percentage returns from trades."""
        returns = get_trade_returns(self.trades)
        self.assertEqual(len(returns), 3)
        self.assertAlmostEqual(returns[0], 0.10)
        self.assertAlmostEqual(returns[1], 0.10)
        self.assertAlmostEqual(returns[2], -0.11)

    def test_run_monte_carlo_basic(self):
        """Test basic MC simulation structural output."""
        # Fix seed for reproducibility in structure checks
        np.random.seed(42)
        returns = [0.1, -0.05, 0.02, 0.08, -0.1]
        mc = run_monte_carlo(returns, initial_capital=100000, simulations=100)

        self.assertEqual(len(mc.final_equity_distribution), 100)
        self.assertEqual(len(mc.drawdown_distribution), 100)
        self.assertEqual(len(mc.sample_equity_curves), 50)  # Capped at 50

        # Equity curves should be length N+1 (includes initial capital)
        self.assertEqual(len(mc.sample_equity_curves[0]), len(returns) + 1)
        self.assertEqual(mc.sample_equity_curves[0][0], 100000.0)

        stats = mc.statistics
        self.assertEqual(stats["n_trades"], 5)
        self.assertEqual(stats["simulations"], 100)
        self.assertIn("median_return", stats)
        self.assertIn("p5_return", stats)

    def test_run_monte_carlo_empty(self):
        """Test edge case with no trades."""
        mc = run_monte_carlo([], initial_capital=100000, simulations=100)
        self.assertEqual(mc.statistics["n_trades"], 0)
        self.assertEqual(mc.statistics["simulations"], 0)
        self.assertEqual(mc.final_equity_distribution, [100000.0])
        self.assertIn("No trades to simulate.", mc.warnings[0])

    def test_fragility_warnings(self):
        """Test warning generation flags."""
        # A guaranteed ruin scenario (e.g. huge string of losses)
        bad_returns = [-0.1] * 10
        mc_bad = run_monte_carlo(bad_returns, initial_capital=100000, simulations=10)
        
        warn_text = " ".join(mc_bad.warnings)
        self.assertIn("negative", warn_text) # Should trigger returning negative warning
        self.assertIn("Worst-case return is", warn_text)

        # A perfect scenario (no losses)
        good_returns = [0.05] * 10
        mc_good = run_monte_carlo(good_returns, initial_capital=100000, simulations=10)
        self.assertEqual(len(mc_good.warnings), 0)


if __name__ == "__main__":
    unittest.main()
