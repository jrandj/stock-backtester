import datetime
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
from matplotlib.dates import num2date, date2num
from mplfinance.original_flavor import candlestick_ochl
import sqlalchemy
from sqlalchemy import MetaData, Table, Column, Integer, String, Float, DateTime, ForeignKey
import config
import math


def import_data(csv_file, hdf_file, path):
    # Import from hdf file if available
    # Else import from csv file and create hdf file
    # Else create CSV file and hd5 file
    # print(csv_file, hdf_file, path)

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
    candlestick_ochl(ax1, candlesticks, width=1, colorup='b', colordown='m')
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


def results_to_csv(path, result):
    # Save results to .csv
    results = [result.ticker, result.Performance.gain, result.Performance.gain_ref,
               result.Performance.annualised_return,
               [pd.to_datetime(i).strftime("%d-%m-%Y") for i in result.data["buy_signal_date"].tolist() if
                not pd.isna(i)], result.buy_transactions, result.buy_transaction_equity,
               pd.to_datetime(result.Performance.start_date).strftime("%d-%m-%Y"), result.Performance.start_price,
               [pd.to_datetime(i).strftime("%d-%m-%Y") for i in result.data["sell_signal_date"].tolist() if
                not pd.isna(i)], result.sell_transactions, result.sell_transaction_equity,
               pd.to_datetime(result.Performance.end_date).strftime("%d-%m-%Y"), result.Performance.end_price,
               result.Performance.annualised_return_ref, result.strategy.required_profit,
               result.strategy.required_volume, result.strategy.required_pct_change_min,
               result.strategy.required_pct_change_max, "Strategy(" + str(result.strategy.required_profit) + ", " + str(
            result.strategy.required_pct_change_min) + ", " + str(result.strategy.required_pct_change_max) + ", " + str(
            result.strategy.required_volume) + ")"]
    schema = ["Ticker", "Strategy Gain", "Buy and Hold Gain", "Annual Return", "Buy Signals",
              "Buy Transactions", "Buy Transaction Equity",
              "Position Start Date", "Position Equity Start", "Sell Signals", "Sell Transactions",
              "Sell Transaction Equity", "Position End Date",
              "Position Equity End", "Buy and Hold Annual Return", "Required Profit", "Required Volume",
              "Required % Change Min", "Required % Change Max", "Strategy"]
    file_exists = os.path.isfile(path + r" Performance.csv")
    with open(path + r" Performance.csv", "a", newline='') as csv_file:
        wr = csv.writer(csv_file, delimiter="~")
        if not file_exists:
            wr.writerow([g for g in schema])
        wr.writerow(results)
        csv_file.close()
    return


def result_to_db(result):
    engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect={}".format(config.params))
    result_dict = result.performance_as_dict()
    result_dict["timestamp"] = datetime.datetime.now()
    df = pd.DataFrame.from_records([result_dict])

    # Write results to performance table
    df.to_sql(config.performance_table, con=engine, chunksize=math.floor(2100 / len(df.columns)), method="multi",
              if_exists="append", index=False)

    # Get foreign key from performance table
    last_id = engine.execute("select MAX(id) from performance;").scalar()
    last_id_buy_array = np.array([last_id] * len(result.buy_transactions)).reshape(len(result.buy_transactions), 1)
    last_id_sell_array = np.array([last_id] * len(result.sell_transactions)).reshape(len(result.sell_transactions), 1)

    # Prepare transaction data for database write
    buy_transactions_array = np.array(result.buy_transactions).reshape(len(result.buy_transactions), 1)
    buy_equity_array = np.array(result.buy_transaction_equity).reshape(len(result.buy_transaction_equity), 1)
    buy_transaction_type = np.array(["buy"] * len(result.buy_transactions)).reshape(len(result.buy_transactions), 1)
    buy_df = pd.DataFrame(
        np.hstack((last_id_buy_array, buy_transaction_type, buy_transactions_array, buy_equity_array)),
        columns=["performance_id", "transaction_type", "transaction_date", "transaction_equity"])
    sell_transactions_array = np.array(result.sell_transactions).reshape(len(result.sell_transactions), 1)
    sell_equity_array = np.array(result.sell_transaction_equity).reshape(len(result.sell_transaction_equity), 1)
    sell_transaction_type = np.array(["sell"] * len(result.sell_transactions)).reshape(len(result.sell_transactions), 1)
    sell_df = pd.DataFrame(
        np.hstack((last_id_sell_array, sell_transaction_type, sell_transactions_array, sell_equity_array)),
        columns=["performance_id", "transaction_type", "transaction_date", "transaction_equity"])

    # Prepare signals data for database write
    buy_signal_array_nonat = result.data.buy_signal_date[~np.isnat(result.data.buy_signal_date)]
    sell_signal_array_nonat = result.data.sell_signal_date[~np.isnat(result.data.sell_signal_date)]

    buy_signals_array = np.array(buy_signal_array_nonat, dtype=np.str).reshape(len(buy_signal_array_nonat), 1)
    buy_signals_type = np.array(["buy"] * len(buy_signals_array)).reshape(len(buy_signals_array), 1)
    last_id_buy_signal_array = np.array([last_id] * len(buy_signals_array)).reshape(len(buy_signals_array), 1)
    sell_signals_array = np.array(sell_signal_array_nonat, dtype=np.str).reshape(len(sell_signal_array_nonat), 1)
    sell_signals_type = np.array(["sell"] * len(sell_signals_array)).reshape(len(sell_signals_array), 1)
    last_id_sell_signal_array = np.array([last_id] * len(sell_signals_array)).reshape(
        len(sell_signals_array), 1)
    buy_signals_df = pd.DataFrame(np.hstack((last_id_buy_signal_array, buy_signals_type, buy_signals_array)),
                                  columns=["performance_id", "signal_type", "signal_date"])
    sell_signals_df = pd.DataFrame(np.hstack((last_id_sell_signal_array, sell_signals_type, sell_signals_array)),
                                   columns=["performance_id", "signal_type", "signal_date"])

    # Write results to transactions table
    buy_df.to_sql(config.transactions_table, con=engine, chunksize=math.floor(2100 / len(df.columns)), method="multi",
                  if_exists="append",
                  index=False)
    sell_df.to_sql(config.transactions_table, con=engine, chunksize=math.floor(2100 / len(df.columns)), method="multi",
                   if_exists="append",
                   index=False)
    buy_signals_df.to_sql(config.signals_table, con=engine, chunksize=math.floor(2100 / len(df.columns)),
                          method="multi", if_exists="append",
                          index=False)
    sell_signals_df.to_sql(config.signals_table, con=engine, chunksize=math.floor(2100 / len(df.columns)),
                           method="multi", if_exists="append",
                           index=False)
    return


def init_performance_table():
    engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect={}".format(config.params))
    meta = MetaData(engine)
    performance = Table(
        "performance", meta,
        Column("id", Integer, primary_key=True),
        Column("ticker", String),
        Column("strategy", String),
        Column("annualised_return", Float),
        Column("annualised_return_ref", Float),
        Column("end_date", DateTime),
        Column("end_price", Float),
        Column("gain", Float),
        Column("gain_ref", Float),
        Column("start_date", DateTime),
        Column("start_price", Float),
        Column("timestamp", DateTime),
    )
    meta.create_all(engine, checkfirst=True)
    return performance


def init_transactions_table(performance):
    engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect={}".format(config.params))
    meta = MetaData()
    transactions = Table(
        "transactions", meta,
        Column("id", Integer, primary_key=True),
        Column("performance_id", Integer, ForeignKey(performance.c.id)),
        Column("transaction_type", String),
        Column("transaction_date", String),
        Column("transaction_equity", String),
    )
    meta.create_all(engine, checkfirst=True)
    return transactions


def init_signals_table(performance):
    engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect={}".format(config.params))
    meta = MetaData()
    signals = Table(
        "signals", meta,
        Column("id", Integer, primary_key=True),
        Column("performance_id", Integer, ForeignKey(performance.c.id)),
        Column("signal_type", String),
        Column("signal_date", String),
    )
    meta.create_all(engine, checkfirst=True)
    return signals
