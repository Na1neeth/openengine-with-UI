from openengine.data.yahoo_connector import YahooFinanceConnector
from openengine.strategies.sample_strategy import SampleStrategy
from openengine.engine.backtester import Backtester
from openengine.utilities.config import INITIAL_CAPITAL

def main():
    # Define parameters for data fetching
    ticker = "RELIANCE.NS"  # Example: Reliance Industries on NSE
    start_date = "2022-01-01"
    end_date = "2023-01-01"

    # Fetch historical data using the YahooFinanceConnector
    data_connector = YahooFinanceConnector()
    data = data_connector.fetch_data(ticker, start_date, end_date, interval="1d")
    
    # Initialize strategy
    strategy = SampleStrategy()
    
    # Create and run the backtester
    backtester = Backtester(data, strategy, initial_capital=INITIAL_CAPITAL)
    portfolio = backtester.run()
    
    print("Backtesting complete. Final portfolio snapshot:")
    print(portfolio.tail())

if __name__ == "__main__":
    main()
