from openengine.execution.order_manager import OrderManager

class LiveTrader:
    def __init__(self, strategy, broker_interface, initial_capital: float = 100000):
        self.strategy = strategy
        self.broker_interface = broker_interface
        self.initial_capital = initial_capital
        self.order_manager = OrderManager()
        self.current_position = 0

    def on_new_data(self, data_point: dict):
        signal = self.strategy.generate_signal_from_data_point(data_point)
        price = data_point["price"]
        timestamp = data_point["timestamp"]

        if signal == 1 and self.current_position <= 0:
            self.current_position = self.order_manager.buy(timestamp, price, self.broker_interface.get_cash())
            # Place a live order via API
            self.broker_interface.place_order(
                strategy="Test Strategy", 
                symbol=data_point.get("symbol", "UNKNOWN"), 
                action="BUY", 
                exchange="NSE"
            )
        elif signal == -1 and self.current_position >= 0:
            self.current_position = self.order_manager.sell(timestamp, price, self.broker_interface.get_cash())
            self.broker_interface.place_order(
                strategy="Test Strategy", 
                symbol=data_point.get("symbol", "UNKNOWN"), 
                action="SELL", 
                exchange="NSE"
            )
