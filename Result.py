import pandas as pd
import numpy as np
import config


class Result:
    """
    A class used to represent a Result.

    Attributes
    ----------
    ticker : sequence
        The stock ticker.
    data : dataframe
        The historical data associated with the ticker.
    strategy : Strategy
        An instance of the Strategy class.
    buy_transactions: sequence
        List of buy transactions.
    sell_transactions: sequence
        List of sell transactions.
    buy_transaction_equity: sequence
        List of equity values corresponding to the buy transactions.
    sell_transaction_equity: sequence
        List of equity values corresponding to the sell transactions.
    Performance : Performance
        An instance of the Performance class.
    transactions : numeric
        The required multiple of the 20D MA volume to generate a buy signal.

    Methods
    -------
    performance_as_dict()
        Returns the performance results in a dictionary.
    tech_indicators()
        Augments the data attribute with columns for technical indicators.
    buy_and_sell_signals()
        Calculate signals where they can be vectorised.
    trade()
        Enters and exit positions based on buy/sell signals.
    calculate_returns()
        Calculate returns after the trade method has been executed.
    print_results()
        Print the performance results to the console.
    """
    def __init__(self, ticker, strategy, raw_data):
        self.ticker = ticker
        self.data = raw_data
        self.strategy = strategy
        self.tech_indicators()
        self.buy_and_sell_signals()
        self.buy_transactions, self.sell_transactions, self.buy_transaction_equity, self.sell_transaction_equity = self.trade()
        self.Performance = self.calculate_returns()
        self.transactions = len(self.buy_transactions + self.sell_transactions)
        self.print_results()

    def performance_as_dict(self):
        """Returns the performance results in a dictionary.

        Parameters
        ----------

        Raises
        ------

        """
        return {'ticker': self.ticker, 'strategy': "Strategy(" + str(self.strategy.required_profit) + ", " + str(
            self.strategy.required_pct_change_min) + ", " + str(self.strategy.required_pct_change_max) + ", " + str(
            self.strategy.required_volume) + ")",
                'annualised_return': self.Performance.annualised_return,
                'annualised_return_ref': self.Performance.annualised_return_ref,
                'end_date': self.Performance.end_date,
                'end_price': self.Performance.end_price,
                'gain': self.Performance.gain,
                'gain_ref': self.Performance.gain_ref,
                'start_date': self.Performance.start_date,
                'start_price': self.Performance.start_price}

    def tech_indicators(self):
        """Augments the data attribute with columns for technical indicators.

        Parameters
        ----------

        Raises
        ------

        """
        self.data = self.data.assign(close_MA_50=self.data[["close"]].ewm(span=50).mean())
        self.data = self.data.assign(close_MA_200=self.data[["close"]].ewm(span=200).mean())
        self.data = self.data.assign(volume_MA_20=self.data[["volume"]].rolling(20).mean())
        self.data = self.data.assign(
            price_change_buy=self.data['close'].pct_change().between(self.strategy.required_pct_change_min,
                                                                     self.strategy.required_pct_change_max))
        self.data = self.data.assign(
            volume_change_buy=(self.data["volume"] > self.strategy.required_volume * self.data["volume_MA_20"]))

        # Money Flow Index (MFI)
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

        # Relative Strength Index (RSI)
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

        # Bollinger Bands
        rolling_mean = self.data[["close"]].ewm(span=50).mean()
        rolling_std = self.data[["close"]].ewm(span=50).std()
        self.data = self.data.assign(BB_upper=rolling_mean + (rolling_std * 2))
        self.data = self.data.assign(BB_lower=rolling_mean - (rolling_std * 2))
        return

    def buy_and_sell_signals(self):
        """Calculate signals where they can be vectorised.

        Generation of sell signal requires iterating through the data which is done in the trade method.

        Parameters
        ----------

        Raises
        ------

        """
        self.data = self.data.assign(buy_signal=np.nan, sell_signal=np.nan, buy_signal_date=np.nan,
                                     sell_signal_date=np.nan)
        buy_prices = self.data["close"].iloc[np.where(self.data["volume_change_buy"] & self.data["price_change_buy"])]
        buy_dates = self.data["date"].iloc[np.where(self.data["volume_change_buy"] & self.data["price_change_buy"])]
        self.data = self.data.assign(buy_signal=buy_prices)
        self.data = self.data.assign(buy_signal_date=buy_dates)
        return

    def trade(self):
        """Enters and exit positions based on buy/sell signals.

        Parameters
        ----------

        Raises
        ------

        """
        buy_transactions, buy_transaction_equity, sell_transactions, sell_transaction_equity = ([] for i in range(4))
        open_long_position, buy_and_hold, buy_and_hold_shares, buy_and_hold, buy_and_hold_shares, shares = (
            0, 0, 0, 0, 0, 0)
        buy_and_hold_position_array, open_long_position_array, strategy_equity_array, buy_and_hold_equity_array = (
            np.full(len(self.data["close"].values), np.nan) for i in range(4))
        # Create buy signal and buy signal dates without NaN or NaT (NaN and NaT inclusive arrays required for plots)
        buy_signal_array_nonan = self.data["buy_signal"].values[~np.isnan(self.data["buy_signal"].values)]
        buy_signal_array_dates_nonat = self.data["buy_signal_date"].values[
            ~np.isnat(self.data["buy_signal_date"].values)]
        j = 0
        cash = config.cash
        buy_and_hold_cash = config.buy_and_hold_cash

        for i in range(0, len(self.data["close"].values)):

            # Handle buy
            if np.isfinite(self.data["buy_signal"].values[i]):
                if not open_long_position:
                    open_long_position = self.data["close"].values[i]
                    shares = (1 - config.transaction_fee) * (cash / open_long_position)
                    cash = 0
                    buy_transactions.append(pd.to_datetime(self.data["date"].values[i]).strftime("%d-%m-%Y"))
                    buy_transaction_equity.append(round(shares * self.data["close"].values[i] + cash, 2))
                if not buy_and_hold:
                    buy_and_hold_shares = ((1 - config.transaction_fee) * buy_and_hold_cash) / \
                                          self.data["close"].values[i]
                    buy_and_hold_cash = 0
                    buy_and_hold = 1

            # Handle sell
            elif (j < len(buy_signal_array_nonan) and self.data["date"].values[i] > buy_signal_array_dates_nonat[j] and
                  self.data["close"].values[
                      i] > self.strategy.required_profit *
                  buy_signal_array_nonan[j]):
                # Need to offset the index which is based on the original dataframe with all tickers
                self.data.at[self.data.index[0] + i, "sell_signal"] = self.data["close"].values[i]
                self.data.at[self.data.index[0] + i, "sell_signal_date"] = pd.to_datetime(self.data["date"].values[i])
                if open_long_position:
                    j = j + 1
                    cash = (1 - config.transaction_fee) * shares * self.data["close"].values[i]
                    shares = 0
                    open_long_position = 0
                    sell_transactions.append(pd.to_datetime(self.data["date"].values[i]).strftime("%d-%m-%Y"))
                    sell_transaction_equity.append(round(shares * self.data["close"].values[i] + cash, 2))

            # Record open positions
            open_long_position_array[i] = self.data["close"].values[i] if open_long_position else 0
            buy_and_hold_position_array[i] = self.data["close"].values[i] if buy_and_hold else 0

            # Record equity
            buy_and_hold_equity_array[i] = buy_and_hold_shares * buy_and_hold_position_array[
                i] + buy_and_hold_cash
            strategy_equity_array[i] = shares * open_long_position_array[i] + cash

        self.data.sell_signal_date = self.data.sell_signal_date.astype("datetime64[ns]", copy=False)
        self.data = self.data.assign(strategy_equity=strategy_equity_array,
                                     buy_and_hold_equity=buy_and_hold_equity_array,
                                     open_long_position=open_long_position_array,
                                     buy_and_hold_position=buy_and_hold_position_array)
        return buy_transactions, sell_transactions, buy_transaction_equity, sell_transaction_equity

    def calculate_returns(self):
        """Calculate returns after the trade method has been executed.

        Parameters
        ----------

        Raises
        ------

        """
        # Calculate returns using strategies and buy and hold
        date_index_long = np.isfinite(self.data["open_long_position"])
        date_index_buy_and_hold = np.isfinite(self.data["buy_and_hold_position"])

        # Handle case where there is no long position
        if self.data["date"][date_index_long].empty:
            performance = Performance(0, 0, 0, 0, 0, 0, 0, 0)
            return performance
        else:
            start_date = self.data["date"][date_index_long].iloc[0]
            start_date_ref = self.data["date"][date_index_buy_and_hold].iloc[0]
            start_price = self.data["strategy_equity"][date_index_long].iloc[0]
            start_price_ref = self.data["buy_and_hold_equity"][date_index_buy_and_hold].iloc[0]
            end_date = self.data["date"][date_index_long].iloc[-1]
            end_date_ref = self.data["date"][date_index_buy_and_hold].iloc[-1]
            end_price = self.data["strategy_equity"][date_index_long].iloc[-1]
            end_price_ref = self.data["buy_and_hold_equity"][date_index_buy_and_hold].iloc[-1]

        # Compute annualised returns
        delta = 1 + (end_date - start_date).days
        delta_ref = 1 + (end_date_ref - start_date_ref).days
        annualised_return = 100 * (((end_price / start_price) ** (365 / delta)) - 1)
        annualised_return_ref = 100 * (((end_price_ref / start_price_ref) ** (365 / delta_ref)) - 1)
        gain = end_price / start_price
        gain_ref = end_price_ref / start_price_ref
        performance = Performance(annualised_return, annualised_return_ref, start_price, start_date, end_price,
                                  end_date, gain, gain_ref)
        return performance

    def print_results(self):
        """Print the performance results to the console.

        Parameters
        ----------

        Raises
        ------

        """
        print(str(self.ticker) + " Strategy Annual Return: " + str(self.Performance.annualised_return) + "%" + "\n" +
              str(self.ticker) + " Buy Signals: " + str(
            [pd.to_datetime(i).strftime("%d-%m-%Y") for i in self.data["buy_signal_date"].tolist() if
             not pd.isna(i)]) + "\n" +
              str(self.ticker) + " Buy Transactions: " + str(self.buy_transactions) + "\n" +
              str(self.ticker) + " Buy Transaction Equity: " + str(self.buy_transaction_equity) + "\n" +
              str(self.ticker) + " Position Start Date: " + str(
            pd.to_datetime(self.Performance.start_date).strftime("%d-%m-%Y")) + "\n" +
              str(self.ticker) + " Position Equity Start: " + str(self.Performance.start_price) + "\n" +
              str(self.ticker) + " Sell Signals: " + str(
            [pd.to_datetime(i).strftime("%d-%m-%Y") for i in self.data["sell_signal_date"].tolist() if
             not pd.isna(i)]) + "\n" +
              str(self.ticker) + " Sell Transactions: " + str(self.sell_transactions) + "\n" +
              str(self.ticker) + " Sell Transaction Equity: " + str(self.sell_transaction_equity) + "\n" +
              str(self.ticker) + " Position End Date: " + str(
            pd.to_datetime(self.Performance.end_date).strftime("%d-%m-%Y")) + "\n" +
              str(self.ticker) + " Position Equity End: " + str(self.Performance.end_price) + "\n" +
              str(self.ticker) + " Buy and Hold Annual Return: " + str(
            self.Performance.annualised_return_ref) + "%" + "\n" +
              str(self.ticker) + " Strategy Gain: " + str(self.Performance.gain) + "\n" +
              str(self.ticker) + " Buy and Hold Gain: " + str(self.Performance.gain))
        return


class Performance:
    """
    A class used to hold the performance for the Result.

    Attributes
    ----------
    annualised_return : numeric
        The annualised return based on equity changes following the buy and sell transactions (based on the trading
        strategy) in the trade method.
    annualised_return_ref : numeric
        The annualised return based on equity changes following the buy and hold transactions in the trade method.
    start_price : numeric
        The equity at the start of the strategy.
    start_date : numeric
        The date at the start of the strategy.
    end_price : numeric
        The equity at the end of the strategy.
    end_date : numeric
        The date at the end of the strategy.
    gain : numeric
        The raw gain (i.e. not annualised) based on equity changes following the buy and sell transactions (based on
        the trading strategy) in the trade method.
    gain_ref : numeric
        The raw gain (i.e. not annualised) based on equity changes following the buy and hold transactions
        in the trade method.

    Methods
    -------

    """
    def __init__(self, annualised_return, annualised_return_ref, start_price, start_date, end_price, end_date, gain,
                 gain_ref):
        self.annualised_return = np.round(annualised_return, 2)
        self.annualised_return_ref = np.round(annualised_return_ref, 2)
        self.start_price = np.round(start_price, 2)
        self.start_date = start_date
        self.end_price = np.round(end_price, 2)
        self.end_date = end_date
        self.gain = np.round(gain, 2)
        self.gain_ref = np.round(gain_ref, 2)
        return
