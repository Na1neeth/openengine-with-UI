# OpenEngine

OpenEngine is a Python library for backtesting and live trading in Indian markets. It features an event-driven architecture and integrates with the OpenAlgo PlaceOrder API for live execution.

## Features

- Event-driven backtesting engine
- Live trading support
- Integration with Yahoo Finance for historical data
- Moving average crossover strategy included as example
- Modular design for easy extension
- Support for Indian markets (NSE)

## Installation

Install from PyPI:

```bash
pip install openengine
```

For development installation:

```bash
git clone https://github.com/marketcalls/openengine.git
cd openengine
pip install -e .
```

## Quick Start

```python
from openengine.data.yahoo_connector import YahooFinanceConnector
from openengine.strategies.sample_strategy import SampleStrategy
from openengine.engine.backtester import Backtester

# Initialize components
data_connector = YahooFinanceConnector()
data = data_connector.fetch_data("RELIANCE.NS", "2023-01-01", "2025-01-01")
strategy = SampleStrategy()

# Run backtest
backtester = Backtester(data, strategy, initial_capital=100000)
portfolio = backtester.run()
print(portfolio.tail())
```

## Usage

### Backtesting

Run the included example:

```bash
python -m openengine.main
```

### Live Trading

```python
from openengine.engine.live_trader import LiveTrader
from openengine.execution.broker_interface import BrokerInterface
from openengine.strategies.sample_strategy import SampleStrategy

# Initialize components
broker = BrokerInterface("http://your-broker-api", "your-api-key")
strategy = SampleStrategy()
trader = LiveTrader(strategy, broker)

# Process market data
trader.on_new_data({
    "symbol": "RELIANCE.NS",
    "price": 2500.0,
    "timestamp": "2025-02-06 09:15:00"
})
```

## Development

### Adding New Strategies

Create a new strategy by inheriting from `BaseStrategy`:

```python
from openengine.strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def generate_signals(self, data):
        # Your strategy logic here
        return signals
```

### Running Tests

```bash
python -m pytest tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the AGPLv3 License - see the [LICENSE](LICENSE) file for details.
