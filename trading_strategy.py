import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import csv


def plot_price(df):
    # Plot prices and equity over time
    plt.style.use('ggplot')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax1 = fig.add_subplot(211)
    ax3 = fig.add_subplot(212)
    ax3.plot(df[["date"]], df[["strategy_equity"]], label="Strategy")
    ax3.plot(df[["date"]], df[["buy_and_hold_equity"]], "k", label="Buy and Hold")
    ax.set(title=df["ticker"].iloc[0] + " closing price and volume")
    ax1.set(ylabel="price")
    ax3.set(xlabel="date", ylabel="equity")
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax1.plot(df[["date"]], df[["close"]], 'b', label="Close")
    ax1.plot(df[["date"]], df[["close_MA_50"]], 'y', label="50EWMA")
    ax1.plot(df[["date"]], df[["close_MA_200"]], 'k', label="200EWMA")
    ax1.plot(df[["date"]], df[["buy_signal"]], 'r*', label="Buy Signal")
    ax1.plot(df[["date"]], df[["sell_signal"]], 'k*', label="Sell Signal")
    sell_dates = df["date"][df["sell_signal"].notnull()]
    buy_dates = df["date"][df["buy_signal"].notnull()]
    for xc in sell_dates:
        ax3.axvline(x=xc, color='k', linestyle='--')
    for xc in buy_dates:
        ax3.axvline(x=xc, color='r', linestyle='--')
    ax1.plot(df[["date"]], df[["open_long_position"]], 'g', label="Open Long Position")
    ax1.plot(df[["date"]], df[["open_short_position"]], 'm', label="Open Short Position")
    ax1.legend(loc="upper right")
    ax3.legend(loc="upper right")
    plt.show()
    return df


def trade(df, active_short):
    # Enter and exit positions based on buy/sell signals
    transaction_fee = 0.011
    open_short_position = 0
    open_long_position = 0
    buy_and_hold = 0
    buy_and_hold_shares = 0
    shares = 0
    cash = 100000
    buy_and_hold_cash = 100000
    df = df.assign(open_short_position=np.nan, open_long_position=np.nan, strategy_equity=np.nan,
                   buy_and_hold_equity=np.nan)

    for i in range(0, df.shape[0]):
        # Enter and exit positions based on buy/sell signals
        if np.isfinite(df["sell_signal"].iloc[i]):
            if open_long_position:
                cash = (1 - transaction_fee) * shares * df["close"].iloc[i]
                open_long_position = 0
            if not open_short_position:
                open_short_position = df["close"].iloc[i]
                shares = cash / open_short_position
                cash = 0
            if not buy_and_hold:
                buy_and_hold_shares = ((1 - transaction_fee) * buy_and_hold_cash) / df["close"].iloc[i]
                buy_and_hold_cash = 0
                buy_and_hold = 1

        if np.isfinite(df["buy_signal"].iloc[i]):
            if open_short_position:
                if active_short:
                    cash = (1 - transaction_fee) * shares * (
                            2 * open_short_position - df["close"].iloc[i])  # start+(start-end)
                else:
                    cash = shares * open_short_position
                open_short_position = 0
            if not open_long_position:
                open_long_position = df["close"].iloc[i]
                shares = (1 - transaction_fee) * (cash / open_long_position)
                cash = 0
            if not buy_and_hold:
                buy_and_hold_shares = ((1 - transaction_fee) * buy_and_hold_cash) / df["close"].iloc[i]
                buy_and_hold_cash = 0
                buy_and_hold = 1

        # Calculate equity based on position
        if open_long_position:
            long_equity = shares * df["close"].iloc[i]
            df.iloc[i, df.columns.get_loc("open_long_position")] = df.iloc[i, df.columns.get_loc("close")]
        else:
            long_equity = 0
        if open_short_position:
            if active_short:
                short_equity = shares * (2 * open_short_position - df["close"].iloc[i])
            else:
                short_equity = shares * open_short_position
            df.iloc[i, df.columns.get_loc("open_short_position")] = df.iloc[i, df.columns.get_loc("close")]
        else:
            short_equity = 0
        df.iloc[i, df.columns.get_loc("strategy_equity")] = long_equity + short_equity + cash
        df.iloc[i, df.columns.get_loc("buy_and_hold_equity")] = buy_and_hold_shares * df["close"].iloc[i] \
            + buy_and_hold_cash
    return df


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
    # Add moving averages to dataframe
    df = df.assign(close_MA_50=df[["close"]].ewm(com=24.5).mean())
    df = df.assign(close_MA_200=df[["close"]].ewm(com=99.5).mean())
    df = df.assign(volume_MA_50=df[["volume"]].rolling(window=50).mean())
    return df


def buy_and_sell_signals(df):
    # Calculate buy and sell signals based on moving average crossover
    df = df.assign(buy_signal=np.nan, sell_signal=np.nan)
    n1 = df["close_MA_50"].shift(1)
    n2 = df["close_MA_200"].shift(1)
    sell_signals = (df["close_MA_50"] < df["close_MA_200"]) & (n1 > n2)
    buy_signals = (df["close_MA_50"] > df["close_MA_200"]) & (n1 < n2)
    df = df.assign(sell_signal=df["close_MA_50"][sell_signals])
    df = df.assign(buy_signal=df["close_MA_50"][buy_signals])
    return df


def import_data(csv_file, hdf_file, path):
    # Import from hdf file if available
    # Else import from csv file and create hdf file
    # Else create CSV file and hd5 file
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

        df = (pd.read_csv(f, names=["date", "open", "close", "volume", "ticker"], parse_dates=["date"],
                          dtype={"ticker": str}, date_parser=dateparse, skiprows=1) for f in all_files)
        df = pd.concat(df, ignore_index=True)
        df.to_csv(os.path.join(path, r"data.csv"), sep=",",
                  header=["date", "open", "close", "volume", "ticker"])
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
    file_exists = os.path.isfile(path + r"performance.csv")
    with open(path + r"performance.csv", "a", newline='') as csv_file:
        wr = csv.writer(csv_file)
        if not file_exists:
            wr.writerow([g for g in schema])
        wr.writerow(results)
        csv_file.close()
    return


def print_results(ticker, transactions, annualised_return, annualised_return_ref, start_price, start_price_ref,
                  end_price, end_price_ref, start_date, start_date_ref, end_date, end_date_ref, start):
    # It feels like bad practise to make a function for this
    print(str(ticker) + " strategy annual return: " + str(annualised_return) + "\n" +
          str(ticker) + " transactions: " + str(transactions) + "\n" +
          str(ticker) + " start price: " + str(start_price) + "\n" +
          str(ticker) + " start price reference: " + str(start_price_ref) + "\n" +
          str(ticker) + " start date: " + str(start_date) + "\n" +
          str(ticker) + " start date ref: " + str(start_date_ref) + "\n" +
          str(ticker) + " end price: " + str(end_price) + "\n" +
          str(ticker) + " end price reference: " + str(end_price_ref) + "\n" +
          str(ticker) + " end date: " + str(end_date) + "\n" +
          str(ticker) + " end date reference: " + str(end_date_ref) + "\n" +
          str(ticker) + " buy and hold annualised return: " + str(annualised_return_ref) + "\n" +
          "Elapsed Time: " + '{0:0.1f} seconds'.format(time.time() - start))
    return


def main():
    csv_file = r"\data.csv"
    path = r"C:\Users\James\Desktop\Historical Data\Converted Data\Equities"
    hdf_file = r"\data.h5"
    tickers = ["ABP", "ABC", "AGL", "ALQ", "ALU", "AWC", "AMC", "AMP", "ANN", "ANZ", "APA", "APO", "ARB", "AAD", "ALL",
               "AHY", "ASX", "AZJ", "ASL", "AST", "API", "AHG", "AOG", "BOQ", "BAP", "BPT", "BGA", "BAL", "BEN", "BHP",
               "BKL", "BSL", "BLD", "BXB", "BRG", "BKW", "BTT", "BWP", "CTX", "CAR", "CGF", "CHC", "CQR", "CNU", "CLW",
               "CIM", "CWY", "CCL", "COH", "CBA", "CPU", "CTD", "CGC", "CCP", "CMW", "CWN", "CSL", "CSR", "CYB", "DXS",
               "DHG", "DMP", "DOW", "DLX", "ECX", "EHE", "EVN", "FXJ", "FPH", "FBU", "FLT", "FMG", "GUD", "GEM", "GXY",
               "GTY", "GMA", "GMG", "GPT", "GNC", "GXL", "GOZ", "GWA", "HVN", "HSO", "ILU", "IPL", "IGO", "IFN", "IAG",
               "IOF", "IVC", "IFL", "IPH", "IRE", "INM", "JHX", "JHG", "JBH", "LLC", "A2M", "LNK", "LYC", "MFG", "MGR",
               "MIN", "MMS", "MND", "MPL", "MQA", "MQG", "MTR", "MTS", "MYO", "MYX", "NAB", "NAN", "NCM", "NEC", "NHF",
               "NSR", "NST", "NUF", "NVT", "NWS", "NXT", "OML", "ORA", "ORE", "ORG", "ORI", "OSH", "OZL", "PGH", "PLS",
               "PMV", "PPT", "PRY", "PTM", "QAN", "QBE", "QUB", "REA", "RFG", "RHC", "RIO", "RMD", "RRL", "RSG", "RWC",
               "S32", "SAR", "SBM", "SCG", "SCP", "SDA", "SDF", "SEK", "SFR", "SGM", "SGP", "SGR", "SHL", "SIG", "SIQ",
               "SKC", "SKI", "SOL", "SPK", "SRX", "STO", "SUL", "SUN", "SVW", "SWM", "SXL", "SYD", "SYR", "TAH", "TCL",
               "TGR", "TLS", "TME", "TNE", "TPM", "TWE", "VCX", "VOC", "VVR", "WBC", "WEB", "WES", "WFD", "WHC", "WOR",
               "WOW", "WPL", "WSA", "WTC", "XRO"]
    active_short = 0
    historical_data = import_data(csv_file, hdf_file, path)

    for ticker in tickers:
        print("Current Ticker: ", ticker)
        start = time.time()
        historical_data_trim = historical_data.loc[historical_data["ticker"] == ticker]
        historical_data_trim = tech_indicators(historical_data_trim)
        historical_data_trim = buy_and_sell_signals(historical_data_trim)
        historical_data_trim = trade(historical_data_trim, active_short)
        annualised_return, annualised_return_ref, start_price, start_price_ref, end_price, end_price_ref, \
            start_date, start_date_ref, end_date, end_date_ref = calculate_returns(historical_data_trim)
        transactions = historical_data_trim["buy_signal"].count() + historical_data_trim["sell_signal"].count()
        print_results(ticker, transactions, annualised_return, annualised_return_ref, start_price, start_price_ref,
            end_price, end_price_ref, start_date, start_date_ref, end_date, end_date_ref, start)
        results_to_csv(path, ticker, annualised_return, transactions, start_price, start_price_ref, end_price,
                       end_price_ref, annualised_return_ref, start_date, start_date_ref, end_date, end_date_ref)
        #plot_price(historical_data_trim)


main()
