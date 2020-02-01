import datetime
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
from matplotlib.dates import num2date, date2num
from mpl_finance import candlestick_ochl


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


def plot_price(df, ticker):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = ax1.twinx()
    ax3 = fig.add_subplot(212)

    # Create synthetic date list to smooth out the gaps in trading days (weekends etc.)
    smoothdate = date2num(df["date"])
    for i in range(len(smoothdate) - 1):
        smoothdate[i + 1] = int(smoothdate[i]) + 1

    # Create candlestick chart
    candlesticks = zip(smoothdate, df["open"], df["close"], df["high"], df["low"], df["volume"])
    candlestick_ochl(ax1, candlesticks, width=1, colorup='g', colordown='r')
    ax1.plot(smoothdate, df["close_MA_50"], "k-", label="Close 50D MA", linewidth=0.5)

    # Add buy and sell signals
    ax1.plot(smoothdate, df["buy_signal"], 'g*', label="Buy Signal")
    ax1.plot(smoothdate, df["sell_signal"], 'r*', label="Sell Signal")

    # Create volume bar chart
    pos = df["open"] - df["close"] < 0
    neg = df["open"] - df["close"] > 0
    ax2.bar(smoothdate[pos], df["volume"][pos], color="green", width=1, align="center")
    ax2.bar(smoothdate[neg], df["volume"][neg], color="red", width=1, align="center")
    ax2.plot(smoothdate, df["volume_MA_20"], "b-", label="Volume 20D MA", linewidth=0.5)

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

    ax1.set_ylabel("Close")
    ax3.set_ylabel("Equity")
    ax1.legend(loc="upper left", prop={"size": 5})
    ax2.legend(loc="upper right", prop={"size": 5})
    ax3.legend(loc="upper left", prop={"size": 5})
    ax1.set_xticklabels(xtick_labels, rotation=45, horizontalalignment="right", fontsize=6)
    ax3.set_xticklabels(xtick_labels, rotation=45, horizontalalignment="right", fontsize=6)
    ax1.title.set_text(ticker)
    plt.tight_layout()
    plt.show()
    return


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
