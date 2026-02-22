# OpenEngine

**OpenEngine** is a powerful, event-driven Python library designed for backtesting and live trading in Indian markets (NSE/BSE). It provides a modular framework for developing trading strategies, running intensive backtests

The project now includes a **Professional Web UI** for interactive analysis and strategy management.

---

## 🚀 Key Features

### Core Engine
- **Event-Driven Architecture**: Simulate real-world trading environments accurately.
- **Yahoo Finance Integration**: Hassle-free historical data fetching for Indian stocks.
- **Built-in Strategies**: Includes standard examples like SMA/EMA crossovers.
- **Performance Metrics**: Detailed stats including Sharpe Ratio, Max Drawdown, CAGR, and Win Rate.

### Web UI
- **Interactive Dashboard**: Modern dark-themed UI built with Flask.
- **Advanced Plotting**: Interactive equity curves and drawdown charts using Plotly.
- **OOS Testing**: Integrated Out-of-Sample testing to prevent overfitting.
- **Monte Carlo Simulation**: Robustness testing for your trading systems.
- **Detailed Trade Logs**: Filterable and searchable records of all simulated trades.

---


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

### Running via Web UI 
Launch the interactive dashboard to run backtests visually:
```bash
python -m openengine.web.app
```
Then open `http://localhost:5000` in your browser.

### Running via CLI
Run the included example script to see the engine in action:
```bash
openengine
```

---

## 🔧 Dependencies
- `pandas` & `numpy` (Data processing)
- `yfinance` (Market data)
- `flask` (Web UI backend)
- `plotly` (Interactive visualizations)

---


---

## 📄 License
This project is licensed under the **AGPLv3 License**. See the `LICENSE` file for more details.

---

*Developed by Navaneeth and OpenEngine Contributors.*
