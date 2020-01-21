import datetime
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
from timeit import default_timer as timer
from matplotlib.dates import num2date, date2num
from mpl_finance import candlestick2_ohlc, candlestick_ochl


def plot_price(df):
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    # Create synthetic date list to smooth out the gaps in trading days (weekends etc.)
    smoothdate = date2num(df["date"])
    for i in range(len(smoothdate) - 1):
        smoothdate[i + 1] = int(smoothdate[i]) + 1

    # Create candlestick chart
    candlesticks = zip(smoothdate, df["open"], df["close"], df["high"], df["low"], df["volume"])
    candlestick_ochl(ax1, candlesticks, width=1, colorup='g', colordown='r')
    ax1.plot(smoothdate, df["close_MA_50"], "k-", label="Close 50D MA", linewidth=0.5)

    # Add buy and sell signals
    ax1.plot(smoothdate, df["buy_signal"], 'b*', label="Buy Signal")
    ax1.plot(smoothdate, df["sell_signal"], 'k*', label="Sell Signal")

    # Create volume bar chart
    pos = df["open"] - df["close"] < 0
    neg = df["open"] - df["close"] > 0
    ax2.bar(smoothdate[pos], df["volume"][pos], color="green", width=1, align="center")
    ax2.bar(smoothdate[neg], df["volume"][neg], color="red", width=1, align="center")
    ax2.plot(smoothdate, df["volume_MA_20"], "k-", label="Volume 20D MA", linewidth=0.5)

    # Add equity chart
    ax3.plot(smoothdate, df["strategy_equity"], "b", label="Strategy")
    ax3.plot(smoothdate, df["buy_and_hold_equity"], "k", label="Buy and Hold")

    # Use smoothed dates for the xticks but the real dates for the tick labels (otherwise the charts appear shrunk)
    actualdate = date2num(df["date"])
    xticks = np.linspace(smoothdate[0], smoothdate[-1], 10)
    xticklabels = pd.date_range(start=num2date(actualdate[0]), end=num2date(actualdate[-1]), periods=10)
    ax1.set_xticks(xticks)
    ax2.set_xticks(xticks)
    ax3.set_xticks(xticks)
    xtick_labels = [datetime.date.isoformat(d) for d in xticklabels]
    ax1.legend(loc="upper left", prop={"size": 5})
    ax2.legend(loc="upper left", prop={"size": 5})
    ax3.legend(loc="upper left", prop={"size": 5})
    ax1.set_xticklabels(xtick_labels, rotation=45, horizontalalignment="right", fontsize=6)
    ax2.set_xticklabels(xtick_labels, rotation=45, horizontalalignment="right", fontsize=6)
    ax3.set_xticklabels(xtick_labels, rotation=45, horizontalalignment="right", fontsize=6)
    plt.tight_layout()
    plt.show()
    return


def trade(df):
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
    sell_signal_array = df["sell_signal"].values
    buy_signal_array = df["buy_signal"].values
    close_array = df["close"].values
    date_array = df["date"].values
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

    df = df.assign(strategy_equity=strategy_equity_array, buy_and_hold_equity=buy_and_hold_equity_array,
                   open_short_position=open_short_position_array, open_long_position=open_long_position_array)
    return df, buy_transactions, sell_transactions


def calculate_returns(df):
    # Calculate returns using strategies and buy and hold
    date_index_long = np.isfinite(df["open_long_position"])
    date_index_short = np.isfinite(df["open_short_position"])

    # Handle cases where there are no buy or sell signals
    a = df["date"][date_index_long]
    b = df["date"][date_index_short]
    if a.empty or b.empty:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    # Short position held first
    if a.index[0] > b.index[0]:
        start_date = b.iloc[0]
        start_price = df["strategy_equity"][date_index_short].iloc[0]
        start_price_ref = df["buy_and_hold_equity"][date_index_short].iloc[0]
    else:
        start_date = a.iloc[0]
        start_price = df["strategy_equity"][date_index_long].iloc[0]
        start_price_ref = df["buy_and_hold_equity"][date_index_long].iloc[0]

    # Long position held last
    if a.index[-1] > b.index[-1]:
        end_date = a.iloc[-1]
        end_price = df["strategy_equity"][date_index_long].iloc[-1]
        end_price_ref = df["buy_and_hold_equity"][date_index_long].iloc[-1]
    else:
        end_date = b.iloc[-1]
        end_price = df["strategy_equity"][date_index_short].iloc[-1]
        end_price_ref = df["buy_and_hold_equity"][date_index_short].iloc[-1]

    # Compute annualised returns
    delta = (end_date - start_date).days
    annualised_return = 100 * (((end_price / start_price) ** (365 / delta)) - 1)
    annualised_return_ref = 100 * (((end_price_ref / start_price_ref) ** (365 / delta)) - 1)
    return annualised_return, annualised_return_ref, start_price, start_price_ref, end_price, end_price_ref, \
           start_date, start_date, end_date, end_date


def tech_indicators(df):
    df = df.assign(close_MA_50=df[["close"]].ewm(span=50).mean())
    df = df.assign(close_MA_200=df[["close"]].ewm(span=200).mean())
    df = df.assign(volume_MA_20=df[["volume"]].rolling(20).mean())
    df = df.assign(price_change_buy=df['close'].pct_change().between(0, 0.05))
    df = df.assign(volume_change_buy=(df["volume"] > 8 * df["volume_MA_20"]))

    # MFI
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    money_flow = typical_price * df["volume"]
    delta = money_flow - money_flow.shift(1)
    delta = pd.Series([0 if np.isnan(x) else x for x in delta])
    positive_money_flow = pd.Series([x if x > 0 else 0 for x in delta])
    negative_money_flow = pd.Series([abs(x) if x < 0 else 0 for x in delta])
    positive_money_flow_sum = positive_money_flow.rolling(window=14).sum().values
    negative_money_flow_sum = negative_money_flow.rolling(window=14).sum().values
    with np.errstate(divide='ignore', invalid='ignore'):
        money_ratio = positive_money_flow_sum / negative_money_flow_sum
    money_flow_index = 100 - 100 / (1 + money_ratio)
    df = df.assign(MFI=money_flow_index)

    # RSI
    delta = df["close"] - df["close"].shift(1)
    delta = pd.Series([0 if np.isnan(x) else x for x in delta])
    up = pd.Series([x if x > 0 else 0 for x in delta])
    down = pd.Series([abs(x) if x < 0 else 0 for x in delta])
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = up.rolling(window=14).mean().values / down.rolling(window=14).mean().values
    relative_strength_index = 100 - 100 / (1 + rs)
    df = df.assign(RSI=relative_strength_index)

    # Stochastic Oscillator
    stochastic_oscillator = pd.Series((df["close"] - df["close"].rolling(window=14, center=False).min()) / (
            df["close"].rolling(window=14, center=False).max() - df["close"].rolling(window=14, center=False).min()))
    stochastic_oscillator = 100 * stochastic_oscillator.rolling(window=3).mean()
    df = df.assign(STO=stochastic_oscillator)

    # Bolinger Bands
    rolling_mean = df[["close"]].ewm(span=50).mean()
    rolling_std = df[["close"]].ewm(span=50).std()
    df = df.assign(BB_upper=rolling_mean + (rolling_std * 2))
    df = df.assign(BB_lower=rolling_mean - (rolling_std * 2))
    return df


def buy_and_sell_signals(df):
    # Calculate buy and sell signals based on moving average crossover
    df = df.assign(buy_signal=np.nan, sell_signal=np.nan)
    n1 = df["close_MA_50"].shift(1)
    n2 = df["close_MA_200"].shift(1)
    buy = df["close"].iloc[np.where(df["volume_change_buy"] & df["price_change_buy"])]
    # sell = df["close"].iloc[np.where(df["close_MA_50"] < 50000000000*df["close_MA_200"])]
    sell = pd.Series({df["close"].index[-1]: df["close"].iloc[-1]})  # dummy sell series with sell at end
    df = df.assign(sell_signal=sell, buy_signal=buy)
    return df


def import_data(csv_file, hdf_file, path):
    # Import from hdf file if available
    # Else import from csv file and create hdf file
    # Else create CSV file and hd5 file
    print(csv_file, hdf_file, path)

    if os.path.isfile(path + hdf_file):
        df = pd.read_hdf(path + hdf_file, 'table')
    elif os.path.isfile(path + csv_file):

        def dateparse(x):
            return pd.datetime.strptime(x, "%Y-%m-%d")

        df = pd.read_csv(path + csv_file, header=0, index_col=0, parse_dates=["date"],
                         date_parser=dateparse)
        df.to_hdf(path + hdf_file, 'table', append=True)
    else:
        all_files = glob.glob(os.path.join(path, "*.csv"))

        def dateparse(x):
            return pd.datetime.strptime(x, "%Y%m%d")

        df = (pd.read_csv(f, names=["date", "open", "high", "low", "close", "volume", "ticker"], parse_dates=["date"],
                          dtype={"ticker": str}, date_parser=dateparse, skiprows=1) for f in all_files)
        df = pd.concat(df, ignore_index=True)
        df.to_csv(os.path.join(path, r"data.csv"), sep=",",
                  header=["date", "open", "high", "low", "close", "volume", "ticker"])
        df.to_hdf(path + hdf_file, 'table', append=True)
    return df


def results_to_csv(path, ticker, annualised_return, transactions, start_price, start_price_ref, end_price,
                   end_price_ref,
                   annualised_return_ref, start_date, start_date_ref, end_date, end_date_ref):
    # Save results to .csv
    results = [ticker, annualised_return, transactions, start_price, end_price, start_date, end_date,
               annualised_return_ref,
               start_price_ref, end_price_ref, start_date_ref, end_date_ref]
    schema = ["Ticker", "Strategy Annual Return", "Strategy Transactions", "Strategy Starting Equity",
              "Strategy Finishing Equity",
              "Strategy Start Date", "Strategy End Date", "Buy&Hold Annual Return", "Buy&Hold Starting Equity",
              "Buy&Hold Finishing Equity", "Buy&Hold Start Date", "Buy&Hold End Date"]
    file_exists = os.path.isfile(path + r" Performance.csv")
    with open(path + r" Performance.csv", "a", newline='') as csv_file:
        wr = csv.writer(csv_file)
        if not file_exists:
            wr.writerow([g for g in schema])
        wr.writerow(results)
        csv_file.close()
    return


def print_results(ticker, transactions, annualised_return, annualised_return_ref):
    print(str(ticker) + " strategy annual return: " + str(annualised_return) + "\n" +
          str(ticker) + " transactions: " + str(transactions) + "\n" +
          str(ticker) + " buy and hold annualised return: " + str(annualised_return_ref))
    return


def main():
    csv_file = r"\data.csv"
    path = r"C:\Users\James\Desktop\Backups\Trading Project\Historical Data\ASX\Equities"
    write_results = 0
    hdf_file = r"\data.h5"
    tickers = ["NOR"]
    first = timer()
    historical_data = import_data(csv_file, hdf_file, path)
    load = timer()
    print("Data import time: " + '{0:0.1f} seconds'.format(load - first))

    for ticker in tickers:
        mask = np.in1d(historical_data['ticker'].values, [ticker])
        historical_data_trim = historical_data[mask]
        historical_data_trim = tech_indicators(historical_data_trim)
        historical_data_trim = buy_and_sell_signals(historical_data_trim)
        historical_data_trim, buy_transactions, sell_transactions = trade(historical_data_trim)
        transactions = len(buy_transactions) + len(sell_transactions)
        annualised_return, annualised_return_ref, start_price, start_price_ref, end_price, end_price_ref, \
        start_date, start_date_ref, end_date, end_date_ref = calculate_returns(historical_data_trim)
        print_results(ticker, transactions, annualised_return, annualised_return_ref)

        if write_results:
            results_to_csv(path, ticker, annualised_return, transactions, start_price, start_price_ref, end_price,
                           end_price_ref, annualised_return_ref, start_date, start_date_ref, end_date, end_date_ref)
        else:
            plot_price(historical_data_trim)

    complete = timer()
    print("Runtime: " + '{0:0.1f} seconds'.format(complete - load))


main()
