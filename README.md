# OpenEngine

**OpenEngine** is a powerful, event-driven Python library designed for backtesting and live trading in Indian markets (NSE/BSE). It provides a modular framework for developing trading strategies, running intensive backtests, and seamlessly transitioning to live execution via the OpenAlgo API.

The project now includes a **Professional Web UI** for interactive analysis and strategy management.

---

## 🚀 Key Features

### Core Engine
- **Event-Driven Architecture**: Simulate real-world trading environments accurately.
- **Yahoo Finance Integration**: Hassle-free historical data fetching for Indian stocks.
- **Built-in Strategies**: Includes standard examples like SMA/EMA crossovers.
- **Performance Metrics**: Detailed stats including Sharpe Ratio, Max Drawdown, CAGR, and Win Rate.
- **Modular Design**: Easy to extend with custom data connectors, brokers, or strategies.

### Web UI
- **Interactive Dashboard**: Modern dark-themed UI built with Flask.
- **Advanced Plotting**: Interactive equity curves and drawdown charts using Plotly.
- **OOS Testing**: Integrated Out-of-Sample testing to prevent overfitting.
- **Monte Carlo Simulation**: Robustness testing for your trading systems.
- **Detailed Trade Logs**: Filterable and searchable records of all simulated trades.

---

## 🛠️ Installation

### 1. Regular Installation (Coming soon to PyPI)
```bash
pip install openengine
```

### 2. Developer Installation (Recommended)
```bash
# Clone the repository
git clone https://github.com/navaneeth/openengine-with-UI.git
cd openengine-with-UI

# Install dependencies
pip install -e .
```

---

## 🚦 Quick Start

### Running via Web UI (Recommended)
Launch the interactive dashboard to run backtests visually:
```bash
openengine-web
```
Then open `http://localhost:5000` in your browser.

### Running via CLI
Run the included example script to see the engine in action:
```bash
openengine
```

---

## 📂 Project Structure

```text
openengine/
├── data/           # Market data connectors (Yahoo Finance, DuckDB)
├── engine/         # Core backtester and live trading logic
├── execution/      # Broker interfaces (OpenAlgo API)
├── strategies/     # Strategy implementations (SMA, EMA, OOS)
├── utilities/      # Math, statistics, and helper functions
└── web/            # Flask application, templates, and static assets
```

---

## 🔧 Dependencies
- `pandas` & `numpy` (Data processing)
- `yfinance` (Market data)
- `duckdb` (Fast local data storage)
- `flask` (Web UI backend)
- `plotly` (Interactive visualizations)

---

## 🤝 Contributing
Contributions are welcome! Whether it's a bug report, a new feature, or documentation improvements:
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License
This project is licensed under the **AGPLv3 License**. See the `LICENSE` file for more details.

---

*Developed by Navaneeth and OpenEngine Contributors.*
