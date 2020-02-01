import pandas as pd
import numpy as np


class Result:
    def __init__(self, ticker, raw_data):
        self.ticker = ticker
        self.data = raw_data
        self.tech_indicators()
        self.buy_and_sell_signals()
        self.buy_transactions, self.sell_transactions = self.trade()
        self.transactions = len(self.buy_transactions + self.sell_transactions)
        self.annualised_return, self.annualised_return_ref, self.start_price, self.start_price_ref, self.end_price, self.end_price_ref, self.start_date, self.end_date = self.calculate_returns()
        self.print_results()

    def tech_indicators(self):
        self.data = self.data.assign(close_MA_50=self.data[["close"]].ewm(span=50).mean())
        self.data = self.data.assign(close_MA_200=self.data[["close"]].ewm(span=200).mean())
        self.data = self.data.assign(volume_MA_20=self.data[["volume"]].rolling(20).mean())
        self.data = self.data.assign(price_change_buy=self.data['close'].pct_change().between(0, 0.05))
        self.data = self.data.assign(volume_change_buy=(self.data["volume"] > 8 * self.data["volume_MA_20"]))

        # MFI
        typical_price = (self.data["high"] + self.data["low"] + self.data["close"]) / 3
        money_flow = typical_price * self.data["volume"]
        delta = money_flow - money_flow.shift(1)
        delta = pd.Series([0 if np.isnan(x) else x for x in delta])
        positive_money_flow = pd.Series([x if x > 0 else 0 for x in delta])
        negative_money_flow = pd.Series([abs(x) if x < 0 else 0 for x in delta])
        positive_money_flow_sum = positive_money_flow.rolling(window=14).sum().values
        negative_money_flow_sum = negative_money_flow.rolling(window=14).sum().values
        with np.errstate(divide='ignore', invalid='ignore'):
            money_ratio = positive_money_flow_sum / negative_money_flow_sum
        money_flow_index = 100 - 100 / (1 + money_ratio)
        self.data = self.data.assign(MFI=money_flow_index)

        # RSI
        delta = self.data["close"] - self.data["close"].shift(1)
        delta = pd.Series([0 if np.isnan(x) else x for x in delta])
        up = pd.Series([x if x > 0 else 0 for x in delta])
        down = pd.Series([abs(x) if x < 0 else 0 for x in delta])
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = up.rolling(window=14).mean().values / down.rolling(window=14).mean().values
        relative_strength_index = 100 - 100 / (1 + rs)
        self.data = self.data.assign(RSI=relative_strength_index)

        # Stochastic Oscillator
        stochastic_oscillator = pd.Series(
            (self.data["close"] - self.data["close"].rolling(window=14, center=False).min()) / (
                    self.data["close"].rolling(window=14, center=False).max() - self.data["close"].rolling(window=14,
                                                                                                           center=False).min()))
        stochastic_oscillator = 100 * stochastic_oscillator.rolling(window=3).mean()
        self.data = self.data.assign(STO=stochastic_oscillator)

        # Bolinger Bands
        rolling_mean = self.data[["close"]].ewm(span=50).mean()
        rolling_std = self.data[["close"]].ewm(span=50).std()
        self.data = self.data.assign(BB_upper=rolling_mean + (rolling_std * 2))
        self.data = self.data.assign(BB_lower=rolling_mean - (rolling_std * 2))
        return

    def buy_and_sell_signals(self):
        # Calculate buy and sell signals based on moving average crossover
        self.data = self.data.assign(buy_signal=np.nan, sell_signal=np.nan)
        n1 = self.data["close_MA_50"].shift(1)
        n2 = self.data["close_MA_200"].shift(1)
        buy_prices = self.data["close"].iloc[np.where(self.data["volume_change_buy"] & self.data["price_change_buy"])]
        buy_dates = self.data["date"].iloc[np.where(self.data["volume_change_buy"] & self.data["price_change_buy"])]

        i = 0
        for row in self.data.itertuples():
            if i < len(buy_prices) and getattr(row, "close") > 1.5 * buy_prices.iloc[i] and getattr(row, "date") > \
                    buy_dates.iloc[i]:
                self.data["sell_signal"].at[getattr(row, "Index")] = getattr(row, "close")
                i = i + 1
        self.data = self.data.assign(buy_signal=buy_prices)
        return

    def trade(self):
        # Enter and exit positions based on buy/sell signals
        buy_transactions = []
        sell_transactions = []
        transaction_fee = 0.011
        open_short_position = 0
        open_long_position = 0
        buy_and_hold = 0
        buy_and_hold_shares = 0
        shares = 0
        cash = 100000
        equity = 0
        buy_and_hold_cash = 100000
        sell_signal_array = self.data["sell_signal"].values
        buy_signal_array = self.data["buy_signal"].values
        close_array = self.data["close"].values
        date_array = self.data["date"].values
        open_long_position_array = np.empty(len(close_array))
        open_long_position_array[:] = np.nan
        open_short_position_array = np.empty(len(close_array))
        open_short_position_array[:] = np.nan
        strategy_equity_array = np.empty(len(close_array))
        strategy_equity_array[:] = np.nan
        buy_and_hold_equity_array = np.empty(len(close_array))
        buy_and_hold_equity_array[:] = np.nan

        for i in range(0, len(close_array)):
            # Enter and exit positions based on buy/sell signals
            if np.isfinite(sell_signal_array[i]):
                if open_long_position:
                    cash = (1 - transaction_fee) * shares * close_array[i]
                    open_long_position = 0
                    sell_transactions.append(date_array[i])
                if not open_short_position:
                    open_short_position = close_array[i]
                    shares = cash / open_short_position
                    cash = 0
                if not buy_and_hold:
                    buy_and_hold_shares = ((1 - transaction_fee) * buy_and_hold_cash) / close_array[i]
                    buy_and_hold_cash = 0
                    buy_and_hold = 1

            if np.isfinite(buy_signal_array[i]):
                if open_short_position:
                    cash = shares * open_short_position
                    buy_transactions.append(date_array[i])
                    open_short_position = 0
                if not open_long_position:
                    open_long_position = close_array[i]
                    shares = (1 - transaction_fee) * (cash / open_long_position)
                    cash = 0
                if not buy_and_hold:
                    buy_and_hold_shares = ((1 - transaction_fee) * buy_and_hold_cash) / close_array[i]
                    buy_and_hold_cash = 0
                    buy_and_hold = 1

            # Calculate equity based on position
            if open_long_position:
                equity = shares * close_array[i]
                open_long_position_array[i] = close_array[i]
            if open_short_position:
                equity = shares * open_short_position
                open_short_position_array[i] = close_array[i]

            strategy_equity_array[i] = equity + cash
            buy_and_hold_equity_array[i] = buy_and_hold_shares * close_array[i] + buy_and_hold_cash

        self.data = self.data.assign(strategy_equity=strategy_equity_array,
                                     buy_and_hold_equity=buy_and_hold_equity_array,
                                     open_short_position=open_short_position_array,
                                     open_long_position=open_long_position_array)
        return buy_transactions, sell_transactions

    def calculate_returns(self):
        # Calculate returns using strategies and buy and hold
        date_index_long = np.isfinite(self.data["open_long_position"])
        date_index_short = np.isfinite(self.data["open_short_position"])

        # Handle cases where there are no buy or sell signals
        a = self.data["date"][date_index_long]
        b = self.data["date"][date_index_short]
        if a.empty or b.empty:
            return 0, 0, 0, 0, 0, 0, 0, 0

        # Short position held first
        if a.index[0] > b.index[0]:
            start_date = b.iloc[0]
            start_price = self.data["strategy_equity"][date_index_short].iloc[0]
            start_price_ref = self.data["buy_and_hold_equity"][date_index_short].iloc[0]
        else:
            start_date = a.iloc[0]
            start_price = self.data["strategy_equity"][date_index_long].iloc[0]
            start_price_ref = self.data["buy_and_hold_equity"][date_index_long].iloc[0]

        # Long position held last
        if a.index[-1] > b.index[-1]:
            end_date = a.iloc[-1]
            end_price = self.data["strategy_equity"][date_index_long].iloc[-1]
            end_price_ref = self.data["buy_and_hold_equity"][date_index_long].iloc[-1]
        else:
            end_date = b.iloc[-1]
            end_price = self.data["strategy_equity"][date_index_short].iloc[-1]
            end_price_ref = self.data["buy_and_hold_equity"][date_index_short].iloc[-1]

        # Compute annualised returns
        delta = (end_date - start_date).days
        annualised_return = 100 * (((end_price / start_price) ** (365 / delta)) - 1)
        annualised_return_ref = 100 * (((end_price_ref / start_price_ref) ** (365 / delta)) - 1)
        return annualised_return, annualised_return_ref, start_price, start_price_ref, end_price, end_price_ref, start_date, end_date

    def print_results(self):
        print(str(self.ticker) + " strategy annual return: " + str(self.annualised_return) + "\n" +
              str(self.ticker) + " transactions: " + str(self.transactions) + "\n" +
              str(self.ticker) + " buy and hold annualised return: " + str(self.annualised_return_ref))
        return
