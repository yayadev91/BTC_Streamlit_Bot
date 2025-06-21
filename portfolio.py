from datetime import datetime

class PaperPortfolio:
    def __init__(self, initial_cash=10000.0):
        self.cash = initial_cash
        self.position = 0
        self.entry_price = None
        self.trade_log = []
        self.equity_curve = []

    def buy(self, price, timestamp):
        self.position = 1
        self.entry_price = price
        self.cash -= price
        self.trade_log.append({"time": timestamp, "type": "BUY", "price": price})

    def sell(self, price, timestamp):
        pnl = price - self.entry_price
        self.cash += price
        self.position = 0
        self.trade_log.append({"time": timestamp, "type": "SELL", "price": price, "pnl": pnl})
        self.entry_price = None

    def update_equity(self, price):
        equity = self.cash + (price if self.position else 0)
        self.equity_curve.append(equity)

    def current_equity(self):
        return self.cash + (self.entry_price if self.position else 0)

    def export_trades_csv(self, path="trades.csv"):
        import pandas as pd
        pd.DataFrame(self.trade_log).to_csv(path, index=False)
