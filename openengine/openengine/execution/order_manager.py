class OrderManager:
    def __init__(self):
        self.position = 0

    def buy(self, timestamp, price, available_cash):
        shares = int(available_cash // price)
        cost = shares * price
        self.position = shares
        print(f"[{timestamp}] BUY: {shares} shares at {price:.2f} (cost: {cost:.2f})")
        return shares, cost

    def sell(self, timestamp, price, available_cash):
        shares = self.position
        proceeds = shares * price
        self.position = 0
        print(f"[{timestamp}] SELL: {shares} shares at {price:.2f} (proceeds: {proceeds:.2f})")
        return 0, proceeds
