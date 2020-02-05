import pandas as pd
import numpy as np


class Result:
    def __init__(self, ticker, strategy, raw_data):
        self.ticker = ticker
        self.data = raw_data
        self.strategy = strategy
        self.tech_indicators()
        self.buy_and_sell_signals()
        self.buy_transactions, self.sell_transactions = self.trade()
        self.Performance = self.calculate_returns()
        self.transactions = len(self.buy_transactions + self.sell_transactions)
        self.print_results()

    def tech_indicators(self):
        self.data = self.data.assign(close_MA_50=self.data[["close"]].ewm(span=50).mean())
        self.data = self.data.assign(close_MA_200=self.data[["close"]].ewm(span=200).mean())
        self.data = self.data.assign(volume_MA_20=self.data[["volume"]].rolling(20).mean())
        self.data = self.data.assign(
            price_change_buy=self.data['close'].pct_change().between(self.strategy.required_pct_change_min,
                                                                     self.strategy.required_pct_change_max))
        self.data = self.data.assign(
            volume_change_buy=(self.data["volume"] > self.strategy.required_volume * self.data["volume_MA_20"]))

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
        self.data = self.data.assign(buy_signal=np.nan, sell_signal=np.nan, buy_signal_date=np.nan,
                                     sell_signal_date=np.nan)
        # n1 = self.data["close_MA_50"].shift(1)
        # n2 = self.data["close_MA_200"].shift(1)
        buy_prices = self.data["close"].iloc[np.where(self.data["volume_change_buy"] & self.data["price_change_buy"])]
        buy_dates = self.data["date"].iloc[np.where(self.data["volume_change_buy"] & self.data["price_change_buy"])]

        i = 0
        for row in self.data.itertuples():
            if i < len(buy_prices) and getattr(row, "close") > self.strategy.required_profit * buy_prices.iloc[
                i] and getattr(row, "date") > buy_dates.iloc[i]:
                self.data.at[getattr(row, "Index"), "sell_signal"] = getattr(row, "close")
                self.data.at[getattr(row, "Index"), "sell_signal_date"] = getattr(row, "date")
                i = i + 1
        self.data = self.data.assign(buy_signal=buy_prices)
        self.data = self.data.assign(buy_signal_date=buy_dates)
        return

    def trade(self):
        # Enter and exit positions based on buy/sell signals
        buy_transactions = []
        sell_transactions = []
        transaction_fee = 0.011
        open_long_position = 0
        buy_and_hold = 0
        buy_and_hold_shares = 0
        shares = 0
        cash = 100000
        buy_and_hold_cash = 100000
        sell_signal_array = self.data["sell_signal"].values
        buy_signal_array = self.data["buy_signal"].values
        close_array = self.data["close"].values
        date_array = self.data["date"].values
        open_long_position_array = np.empty(len(close_array))
        open_long_position_array[:] = np.nan
        strategy_equity_array = np.empty(len(close_array))
        strategy_equity_array[:] = np.nan
        buy_and_hold_equity_array = np.empty(len(close_array))
        buy_and_hold_equity_array[:] = np.nan

        for i in range(0, len(close_array)):
            # Enter and exit positions based on buy/sell signals
            if np.isfinite(sell_signal_array[i]):
                if open_long_position:
                    cash = (1 - transaction_fee) * shares * close_array[i]
                    shares = 0
                    open_long_position = 0
                    sell_transactions.append(pd.to_datetime(date_array[i]).strftime("%Y-%m-%d"))

            if np.isfinite(buy_signal_array[i]):
                if not open_long_position:
                    open_long_position = close_array[i]
                    shares = (1 - transaction_fee) * (cash / open_long_position)
                    buy_transactions.append(pd.to_datetime(date_array[i]).strftime("%Y-%m-%d"))
                    cash = 0
                if not buy_and_hold:
                    buy_and_hold_shares = ((1 - transaction_fee) * buy_and_hold_cash) / close_array[i]
                    buy_and_hold_cash = 0
                    buy_and_hold = 1

            # Calculate equity based on position
            equity = shares * close_array[i]
            # Record when we held an open long position
            if open_long_position:
                open_long_position_array[i] = close_array[i]

            strategy_equity_array[i] = equity + cash
            buy_and_hold_equity_array[i] = buy_and_hold_shares * close_array[i] + buy_and_hold_cash

        self.data = self.data.assign(strategy_equity=strategy_equity_array,
                                     buy_and_hold_equity=buy_and_hold_equity_array,
                                     open_long_position=open_long_position_array)
        return buy_transactions, sell_transactions

    def calculate_returns(self):
        # Calculate returns using strategies and buy and hold
        date_index_long = np.isfinite(self.data["open_long_position"])

        # Handle case where there is no long position
        if self.data["date"][date_index_long].empty:
            performance = Performance(0, 0, 0, 0, 0, 0)
            return performance
        else:
            start_date = self.data["date"][date_index_long].iloc[0]
            start_price = self.data["strategy_equity"][date_index_long].iloc[0]
            start_price_ref = self.data["buy_and_hold_equity"][date_index_long].iloc[0]
            end_date = self.data["date"][date_index_long].iloc[-1]
            end_price = self.data["strategy_equity"][date_index_long].iloc[-1]
            end_price_ref = self.data["buy_and_hold_equity"][date_index_long].iloc[-1]

        # Compute annualised returns
        delta = (end_date - start_date).days
        annualised_return = 100 * (((end_price / start_price) ** (365 / delta)) - 1)
        annualised_return_ref = 100 * (((end_price_ref / start_price_ref) ** (365 / delta)) - 1)
        performance = Performance(annualised_return, annualised_return_ref, start_price, start_date, end_price,
                                  end_date)
        return performance

    def print_results(self):
        print(str(self.ticker) + " Strategy Annual Return: " + str(self.Performance.annualised_return) + "%" + "\n" +
              str(self.ticker) + " Buy Signals: " + str(
            [pd.to_datetime(i).strftime("%Y-%m-%d") for i in self.data["buy_signal_date"].tolist() if
             not pd.isna(i)]) + "\n" +
              str(self.ticker) + " Buy Transactions: " + str(self.buy_transactions) + "\n" +
              str(self.ticker) + " Position Start Date: " + str(
            pd.to_datetime(self.Performance.start_date).strftime("%Y-%m-%d")) + "\n" +
              str(self.ticker) + " Position Equity Start: " + str(self.Performance.start_price) + "\n" +
              str(self.ticker) + " Sell Signals: " + str(
            [pd.to_datetime(i).strftime("%Y-%m-%d") for i in self.data["sell_signal_date"].tolist() if
             not pd.isna(i)]) + "\n" +
              str(self.ticker) + " Sell Transactions: " + str(self.sell_transactions) + "\n" +
              str(self.ticker) + " Position End Date: " + str(
            pd.to_datetime(self.Performance.end_date).strftime("%Y-%m-%d")) + "\n" +
              str(self.ticker) + " Position Equity End: " + str(self.Performance.end_price) + "\n" +
              str(self.ticker) + " Buy and Hold Annual Return: " + str(self.Performance.annualised_return_ref) + "%")
        return


class Performance:
    def __init__(self, annualised_return, annualised_return_ref, start_price, start_date, end_price, end_date):
        self.annualised_return = annualised_return
        self.annualised_return_ref = annualised_return_ref
        self.start_price = start_price
        self.start_date = start_date
        self.end_price = end_price
        self.end_date = end_date
        return
