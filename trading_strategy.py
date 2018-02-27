import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time


class portfolio:
    # Portfolio details
    def __init__(self):
        self.cash = 100000
        self.shares = 0
        self.equity = []
        self.buy_and_hold = []


def plot_price(df, p):
    # Plot stuff
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax1 = fig.add_subplot(211)
    ax3 = fig.add_subplot(212)
    ax1.plot(df[["date"]], df[["close"]], label="Close")
    ax3.plot(df[["date"]], p.equity, label="Strategy")
    ax3.plot(df[["date"]], p.buy_and_hold, "k", label="Buy and Hold")
    ax3.legend(loc="upper right")
    ax.set(title=df["ticker"].iloc[0]+" closing price and volume")
    ax1.set(ylabel="price")
    ax3.set(xlabel="date", ylabel="equity")
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax1.plot(df[["date"]], df[["close_MA_50"]], label="50MA")
    ax1.plot(df[["date"]], df[["close_MA_200"]], 'k', label="200MA")
    ax1.plot(df[["date"]], df[["buy_signal"]], 'r*', label="Buy Signal")
    ax1.plot(df[["date"]], df[["sell_signal"]], 'k*', label="Sell Signal")
    ax1.plot(df[["date"]], df[["open_position"]], 'g', label="Open Position")
    ax1.legend(loc="upper right")
    plt.show()

    # volume figure
    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(111)
    # ax2.set(ylabel="volume")
    # ax2.plot(df[["date"]], df[["volume"]])
    # ax2.plot(df[["date"]], df[["volume_MA_50"]])
    # plt.show()
    return df


def buy_signals_combination(df):
    # Custom strategy
    # Generate buy signal when volume > x*volume moving average and price increase < y% moving price average
    # x = 5, y = 1.1
    x = 5
    y = 1.1
    df = df.assign(buy_signal=np.nan)
    condition = np.logical_and(df["volume"] >= x*df["volume_MA_50"], df["close"] <= y*df["close_MA_50"])
    df["buy_signal"] = np.where(condition, df["close"], np.nan)
    return df


def sell_signals_combination(df):
    # Generate sell signal when x% below ATH
    x = 0.7
    all_time_high = 0
    open_position = 0
    df = df.assign(sell_signal=np.nan)
    for i in range(0, df.shape[0]):
        if np.isfinite(df["buy_signal"].iloc[i]) and open_position == 0:
            all_time_high = df["close"].iloc[i]
            open_position = 1
        # Keep track of ATH
        if open_position == 1:
            if df["close"].iloc[i] > all_time_high:
                all_time_high = df["close"].iloc[i]
            # If x% down from ATH then sell
            if df["close"].iloc[i] < (x * all_time_high):
                df["sell_signal"].iloc[i] = df["close"].iloc[i]
                open_position = 0
    return df


def signals_momentum(df):
    # Generate buy and sell based on momentum crossovers
    df = df.assign(buy_signal=np.nan)
    df = df.assign(sell_signal=np.nan)
    n1 = df["close_MA_50"].shift(1)
    n2 = df["close_MA_200"].shift(1)
    sell_signals = (df["close_MA_50"] <= df["close_MA_200"]) & (n1 >= n2)
    buy_signals = (df["close_MA_50"] >= df["close_MA_200"]) & (n1 <= n2)
    df["sell_signal"][sell_signals] = df["close_MA_50"][sell_signals]
    df["buy_signal"][buy_signals] = df["close_MA_50"][buy_signals]
    return df


def sell_signals_momentum(df):
    # Generate sell signal when price =< x*ATH
    x = 0.7
    all_time_high = 0
    open_position = 0
    df = df.assign(sell_signal=np.nan)

    for i in range(0, df.shape[0]):
        if np.isfinite(df["buy_signal"].iloc[i]) and open_position == 0:
            all_time_high = df["close"].iloc[i]
            open_position = 1
        # Keep track of ATH
        if open_position == 1:
            if df["close"].iloc[i] > all_time_high:
                all_time_high = df["close"].iloc[i]
            # If x% down from ATH then sell
            if df["close"].iloc[i] < (x * all_time_high):
                df["sell_signal"].iloc[i] = df["close"].iloc[i]
                open_position = 0
    return df


def trade(df, p):
    # If a buy signal exists, use all portfolio cash to buy
    # Ignore subsequent buy signals unless a buy has been closed with a sell
    # If a sell signal exists, sell all shares
    # x% trading fee
    x = 0.0011
    open_position = 0
    buy_and_hold = 0
    initial_shares = 0
    initial_cash = p.cash
    df = df.assign(open_position=np.nan)

    for i in range(0, df.shape[0]):
        if open_position:
            # Track portfolio performance
            df["open_position"].iloc[i] = df["close"].iloc[i] #slow
            # need to rework this for short selling as well
            if np.isfinite(df["sell_signal"].iloc[i]):
                p.cash = (1 - x) * (p.shares * df["close"].iloc[i])
                p.shares = 0
                open_position = 0
        else:
            # If we have a buy signal
            if np.isfinite(df["buy_signal"].iloc[i]):
                p.shares = (1 - x) * (p.cash / df["close"].iloc[i])
                p.cash = 0
                open_position = 1
                if not buy_and_hold:
                    initial_shares = p.shares
                    initial_cash = 0
                    buy_and_hold = 1
        p.buy_and_hold.append(initial_shares*df["close"].iloc[i]+initial_cash)
        p.equity.append(p.shares*df["close"].iloc[i]+p.cash)
    return df, p


def load(csv_file, hdf_file, path):
    # Import from hdf file if available
    # Else import from csv file and create hdf file
    # Else create CSV file and hd5 file
    if os.path.isfile(path+hdf_file):
        df = pd.read_hdf(path+hdf_file, 'table')
    elif os.path.isfile(path+csv_file):
        dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
        df = pd.read_csv(path + csv_file, header=0, index_col=0, parse_dates=["date"],
                                      date_parser=dateparse)
        df.to_hdf(path+hdf_file, 'table', append=True)
    else:
        all_files = glob.glob(os.path.join(path, "*.txt"))
        dateparse = lambda x: pd.datetime.strptime(x, '%Y%m%d')
        df = (pd.read_csv(f, names=["ticker", "date", "open", "high", "low", "close", "volume"], parse_dates=["date"],
                          date_parser=dateparse) for f in all_files)
        df = pd.concat(df, ignore_index=True)
        df.to_csv(os.path.join(path, r"historical_data.csv"), sep=",",
                               header=["ticker", "date", "open", "high", "low", "close", "volume"])
        df.to_hdf(path+hdf_file, 'table', append=True)
    return df


def calculate_returns(df):
    # Calculate returns using strategies and holding after buying with initial buy signal
    # Get open position price and date
    a = df["open_position"][np.isfinite(df["open_position"])]
    start_price = df["close"][a.index[0]]
    end_price = df["close"][a.index[-1]]
    b = df["date"][np.isfinite(df["open_position"])]
    start_date = df["date"][b.index[0]]
    end_date = df["date"][b.index[-1]]
    delta = (end_date - start_date).days

    # Get buy and hold prices and dates based on first buy signal and last sell signal
    start_price_ref = df["close"][np.isfinite(df["buy_signal"]).index[0]]
    end_price_ref = df["close"][np.isfinite(df["sell_signal"]).index[-1]]
    start_date_ref = df["date"][np.isfinite(df["buy_signal"]).index[0]]
    end_date_ref = df["date"][np.isfinite(df["sell_signal"]).index[-1]]
    delta_ref = (end_date_ref-start_date_ref).days

    # Compute annualised returns
    annualised_return = 100*(((end_price/start_price)**(365/delta))-1)
    annualised_return_ref = 100*(((end_price_ref/start_price_ref)**(365 / delta_ref)) - 1)
    return annualised_return, annualised_return_ref


def prepare_data(df, ticker):
    # Filter dataframe for ticker and add moving averages and buy/sell signals
    df = df.loc[df["ticker"] == ticker]
    df["close_MA_50"] = df[["close"]].rolling(window=50).mean()
    df["volume_MA_50"] = df[["volume"]].rolling(window=50).mean()
    df["close_MA_200"] = df[["close"]].rolling(window=200).mean()
    # historical_data["close_MA_50"] = historical_data[["close"]].ewm(com=0, min_periods=50).mean()
    # historical_data["volume_MA_50"] = historical_data[["volume"]].ewm(com=0, min_periods=50).mean()
    # historical_data["close_MA_200"] = historical_data[["close"]].ewm(com=0, min_periods=200).mean()
    df = signals_momentum(df)
    #historical_data = buy_signals_combination(historical_data)
    #historical_data = sell_signals_combination(historical_data)
    return df


def main():
    csv_file = r"\historical_data.csv"
    path = r"C:\Users\James\Desktop\Historical Data"
    hdf_file = r"\data.h5"
    tickers = ["FLT"]
    #tickers = ["AMP", "ANZ", "BHP", "BXB", "CBA", "CSL", "IAG", "MQG", "NAB", "QBE", "RIO", "SCG", "SUN", "TLS", "TCL",
    #           "WES", "WFD", "WBC", "WPL", "WOW"]
    historical_data = load(csv_file, hdf_file, path)
    annualised_returns = []
    annualised_returns_ref = []

    for i in range(0, len(tickers)):
        start = time.time()
        myportfolio = portfolio()
        historical_data_trim = prepare_data(historical_data, tickers[i])
        historical_data_trim, myportfolio = trade(historical_data_trim, myportfolio)
        annualised_return, annualised_return_ref = calculate_returns(historical_data_trim)
        annualised_returns.append(annualised_return)
        annualised_returns_ref.append(annualised_return_ref)
        print(str(tickers[i]) + " strategy annual return: " + str(annualised_return)
              + "\n", str(tickers[i]) + " buy and hold annualised return: " +
              str(annualised_return_ref) + "\n" + '{0:0.1f} seconds'.format(time.time() - start))
        plot_price(historical_data_trim, myportfolio)


main()