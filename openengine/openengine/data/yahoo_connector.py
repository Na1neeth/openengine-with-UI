import yfinance as yf

class YahooFinanceConnector:
    def __init__(self):
        pass

    def fetch_data(self, ticker: str, start_date: str, end_date: str, interval: str = "1d"):
        """
        Fetch historical data using yfinance. For Indian stocks, you might need to append '.NS'
        to the ticker (e.g., 'RELIANCE.NS').
        """
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        # Flatten multi-level columns (newer yfinance returns e.g. ("Close", "RELIANCE.NS"))
        if hasattr(data.columns, 'levels') and len(data.columns.levels) > 1:
            data.columns = data.columns.get_level_values(0)
        return data
