import requests

class BrokerInterface:
    def __init__(self, base_url, apikey):
        self.base_url = base_url  # e.g., "http://127.0.0.1:5000"
        self.apikey = apikey
        self.cash = 100000

    def get_cash(self):
        return self.cash

    def place_order(self, strategy, symbol, action, exchange, product="MIS", pricetype="MARKET",
                    quantity="1", price="0", trigger_price="0", disclosed_quantity="0"):
        endpoint = f"{self.base_url}/api/v1/placeorder"
        payload = {
            "apikey": self.apikey,
            "strategy": strategy,
            "exchange": exchange,
            "symbol": symbol,
            "action": action,
            "product": product,
            "pricetype": pricetype,
            "quantity": quantity,
            "price": price,
            "trigger_price": trigger_price,
            "disclosed_quantity": disclosed_quantity
        }
        print("Placing order:", payload)
        try:
            response = requests.post(endpoint, json=payload)
            if response.status_code == 200:
                result = response.json()
                print("Order Response:", result)
                return result
            else:
                print("Error placing order:", response.text)
        except Exception as e:
            print("Exception during order placement:", e)
