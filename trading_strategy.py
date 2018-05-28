import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
from timeit import default_timer as timer


def plot_price(df, buy_transactions, sell_transactions):
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
    # ax1.plot(df[["date"]], df[["BB_upper"]], 'k', label="BB Upper")
    # ax1.plot(df[["date"]], df[["BB_lower"]], 'k', label="BB Lower")
    ax1.plot(df[["date"]], df[["close_MA_200"]], 'k', label="200EWMA")
    ax1.plot(df[["date"]], df[["buy_signal"]], 'r*', label="Buy Signal")
    ax1.plot(df[["date"]], df[["sell_signal"]], 'k*', label="Sell Signal")
    for xc in sell_transactions:
        ax3.axvline(x=xc, color='k', linestyle='--')
    for xc in buy_transactions:
        ax3.axvline(x=xc, color='r', linestyle='--')
    ax1.plot(df[["date"]], df[["open_long_position"]], 'g', label="Open Long Position")
    ax1.plot(df[["date"]], df[["open_short_position"]], 'm', label="Open Short Position")
    ax1.legend(loc="upper right")
    ax3.legend(loc="upper right")
    plt.show()
    return df


def trade(df, active_short):
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
                if active_short:
                    cash = (1 - transaction_fee) * shares * (
                            2 * open_short_position - close_array[i])  # start+(start-end)
                else:
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
            if active_short:
                equity = shares * (2 * open_short_position - df["close"].iloc[i])
            else:
                equity = shares * open_short_position
            open_short_position_array[i] = close_array[i]

        strategy_equity_array[i] = equity + cash
        buy_and_hold_equity_array[i] = buy_and_hold_shares * close_array[i] + buy_and_hold_cash

    df = df.assign(strategy_equity=strategy_equity_array, buy_and_hold_equity=buy_and_hold_equity_array, open_short_position=open_short_position_array, open_long_position=open_long_position_array)
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
    df = df.assign(close_MA_50=df[["close"]].ewm(span=50).mean())  # 24.5 = 50 day
    df = df.assign(close_MA_200=df[["close"]].ewm(span=200).mean())  # 99.5 = 200 day

    # MFI
    typical_price = (df["high"]+df["low"]+df["close"])/3
    money_flow = typical_price*df["volume"]
    delta = money_flow-money_flow.shift(1)
    delta = pd.Series([0 if np.isnan(x) else x for x in delta])
    positive_money_flow = pd.Series([x if x > 0 else 0 for x in delta])
    negative_money_flow = pd.Series([abs(x) if x < 0 else 0 for x in delta])
    positive_money_flow_sum = positive_money_flow.rolling(window=14).sum().values
    negative_money_flow_sum = negative_money_flow.rolling(window=14).sum().values
    with np.errstate(divide='ignore', invalid='ignore'):
        money_ratio = positive_money_flow_sum/negative_money_flow_sum
    money_flow_index = 100 - 100/(1+money_ratio)
    df = df.assign(MFI=money_flow_index)

    # RSI
    delta = df["close"]-df["close"].shift(1)
    delta = pd.Series([0 if np.isnan(x) else x for x in delta])
    up = pd.Series([x if x > 0 else 0 for x in delta])
    down = pd.Series([abs(x) if x < 0 else 0 for x in delta])
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = up.rolling(window=14).mean().values/down.rolling(window=14).mean().values
    relative_strength_index = 100 - 100/(1+rs)
    df = df.assign(RSI=relative_strength_index)

    # Stochastic Oscillator (incomplete)
    stochastic_oscillator = pd.Series((df["close"] - df["close"].rolling(window=14, center=False).min()) / (
        df["close"].rolling(window=14, center=False).max() - df["close"].rolling(window=14, center=False).min()))
    stochastic_oscillator = 100*stochastic_oscillator.rolling(window=3).mean()
    df = df.assign(STO=stochastic_oscillator)

    # Bolinger Bands
    rolling_mean = df[["close"]].ewm(span=50).mean() #24.5 = 50 day
    rolling_std = df[["close"]].ewm(span=50).std()
    df = df.assign(BB_upper=rolling_mean + (rolling_std*2))
    df = df.assign(BB_lower=rolling_mean - (rolling_std*2))

    # OBV
    close_array = df["close"].values
    on_balance_volume = [0] * len(close_array)
    for i in range(0, len(close_array)):
        if i-1 == -1:
            on_balance_volume[i-1] = 0
        elif close_array[i] > close_array[i-1]:
            on_balance_volume[i] = on_balance_volume[i-1] + df.iloc[i, df.columns.get_loc("volume")]
        elif close_array[i] < close_array[i-1]:
            on_balance_volume[i] = on_balance_volume[i - 1] - df.iloc[i, df.columns.get_loc("volume")]
        else:
            on_balance_volume[i] = on_balance_volume[i-1]
    df = df.assign(OBV=(on_balance_volume/df["volume"]))
    return df


def buy_and_sell_signals(df):
    # Calculate buy and sell signals based on moving average crossover
    df = df.assign(buy_signal=np.nan, sell_signal=np.nan)
    n1 = df["close_MA_50"].shift(1)
    n2 = df["close_MA_200"].shift(1)
    # OBV_high = df["OBV"].ewm(span=200).mean()
    # OBV_low = df["OBV"].ewm(span=50).mean()
    buy = df["close"].iloc[np.where((df["close_MA_50"] > df["close_MA_200"]) & (n1 < n2))]
    sell = df["close"].iloc[np.where((df["close_MA_50"] < df["close_MA_200"]) & (n1 > n2))]
    df = df.assign(sell_signal=sell, buy_signal=buy)
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
          "Elapsed Time: " + '{0:0.1f} seconds'.format(timer() - start))
    return


def main():
    csv_file = r"\data.csv"
    path = r"C:\Users\James\Desktop\Historical Data\ASX\Equities"
    hdf_file = r"\data.h5"
    tickers = ["CBA"]
    # all ords 2018
    # tickers = ['3PL', 'A2M', 'AAC', 'AAD', 'ABA', 'ABC', 'ABP', 'ADA', 'ADH', 'AFG', 'AGG', 'AGI', 'AGL', 'AGO', 'AGY',
    #            'AHG', 'AHY', 'AIA', 'AJD', 'AJL', 'AJM', 'AKP', 'ALK', 'ALL', 'ALQ', 'ALU', 'AMA', 'AMC', 'AMI', 'AMP',
    #            'ANN', 'ANZ', 'AOF', 'AOG', 'APA', 'APE', 'API', 'APO', 'APT', 'APX', 'AQG', 'AQZ', 'ARB', 'ARF', 'ARV',
    #            'ASB', 'ASG', 'ASL', 'AST', 'ASX', 'ATL', 'ATS', 'AUB', 'AUZ', 'AVB', 'AVJ', 'AVN', 'AVZ', 'AWC', 'AX1',
    #            'AXP', 'AYS', 'AZJ', 'BAL', 'BAP', 'BBN', 'BDR', 'BEN', 'BFG', 'BGA', 'BHP', 'BIN', 'BKL', 'BKW', 'BKY',
    #            'BLA', 'BLD', 'BLX', 'BLY', 'BNO', 'BOQ', 'BPT', 'BRG', 'BRL', 'BRN', 'BSA', 'BSE', 'BSL', 'BUB', 'BUL',
    #            'BVS', 'BWP', 'BWX', 'BXB', 'CAB', 'CAJ', 'CAN', 'CAR', 'CAT', 'CBA', 'CCL', 'CCP', 'CCV', 'CDA', 'CDD',
    #            'CDP', 'CDV', 'CEN', 'CGC', 'CGF', 'CGL', 'CHC', 'CIA', 'CII', 'CIM', 'CIP', 'CKF', 'CL1', 'CLH', 'CLQ',
    #            'CLW', 'CMA', 'CMW', 'CNI', 'CNU', 'COE', 'COH', 'CPU', 'CQR', 'CRD', 'CRR', 'CSL', 'CSR', 'CTD', 'CTX',
    #            'CUV', 'CVC', 'CVW', 'CWN', 'CWP', 'CWY', 'CYB', 'CZZ', 'DCG', 'DCN', 'DDR', 'DFM', 'DHG', 'DLX', 'DMP',
    #            'DNA', 'DNK', 'DOW', 'DTL', 'DWS', 'DXS', 'ECX', 'EDE', 'EHE', 'EHL', 'ELD', 'EML', 'ENN', 'EOS', 'EPW',
    #            'EQT', 'ERA', 'ERF', 'ESV', 'EVN', 'EVT', 'EWC', 'EXP', 'EZL', 'FAR', 'FBR', 'FBU', 'FDM', 'FET', 'FID',
    #            'FLC', 'FLK', 'FLN', 'FLT', 'FMG', 'FMS', 'FNP', 'FPH', 'FRI', 'FSA', 'FWD', 'FXJ', 'FXL', 'GCS', 'GCY',
    #            'GDF', 'GDI', 'GEM', 'GMA', 'GMG', 'GNC', 'GNG', 'GOR', 'GOW', 'GOZ', 'GPT', 'GRR', 'GSC', 'GSW', 'GTN',
    #            'GTY', 'GUD', 'GWA', 'GXL', 'GXY', 'HAS', 'HFR', 'HLO', 'HOM', 'HPI', 'HRR', 'HSN', 'HSO', 'HT1', 'HTA',
    #            'HUB', 'HUO', 'HVN', 'IAG', 'IDR', 'IDX', 'IEL', 'IFL', 'IFM', 'IFN', 'IGL', 'IGO', 'ILU', 'IMD', 'IMF',
    #            'INA', 'ING', 'INM', 'IOF', 'IPD', 'IPH', 'IPL', 'IRE', 'IRI', 'ISD', 'ISU', 'IVC', 'JBH', 'JHC', 'JHG',
    #            'JHX', 'JIN', 'JLG', 'KAR', 'KDR', 'KGN', 'KMD', 'KSC', 'LEP', 'LIC', 'LLC', 'LNG', 'LNK', 'LOV', 'LVH',
    #            'LYC', 'LYL', 'MAH', 'MAQ', 'MDC', 'MDL', 'MFG', 'MGR', 'MGX', 'MHJ', 'MIN', 'MLB', 'MLD', 'MLX', 'MMI',
    #            'MMS', 'MND', 'MNF', 'MNS', 'MNY', 'MOC', 'MOE', 'MP1', 'MPL', 'MQA', 'MQG', 'MRM', 'MRN', 'MSB', 'MTO',
    #            'MTR', 'MTS', 'MVF', 'MVP', 'MWY', 'MYO', 'MYR', 'MYS', 'MYX', 'NAB', 'NAN', 'NBL', 'NCK', 'NCM', 'NCZ',
    #            'NEA', 'NEC', 'NEU', 'NEW', 'NGI', 'NHC', 'NHF', 'NMT', 'NSR', 'NST', 'NTC', 'NUF', 'NVL', 'NVT', 'NWH',
    #            'NWL', 'NWS', 'NXT', 'NZM', 'OCL', 'OFX', 'OGC', 'OMH', 'OML', 'ONT', 'ORA', 'ORE', 'ORG', 'ORI', 'OSH',
    #            'OVH', 'OZL', 'PAC', 'PAN', 'PCG', 'PDL', 'PEA', 'PEAR', 'PFP', 'PGC', 'PGH', 'PHI', 'PLG', 'PLS', 'PME',
    #            'PMP', 'PMV', 'PNC', 'PNI', 'PNR', 'PNV', 'PPC', 'PPG', 'PPH', 'PPS', 'PPT', 'PRU', 'PRY', 'PSI', 'PSQ',
    #            'PTM', 'PWH', 'QAN', 'QBE', 'QIP', 'QMS', 'QUB', 'RBL', 'RCR', 'REA', 'REG', 'REH', 'REX', 'RFF', 'RFG',
    #            'RHC', 'RHL', 'RIC', 'RIO', 'RKN', 'RMD', 'RMS', 'RND', 'RRL', 'RSG', 'RUL', 'RVA', 'RWC', 'S32', 'SAR',
    #            'SBM', 'SCG', 'SCO', 'SCP', 'SDA', 'SDF', 'SDG', 'SEH', 'SEK', 'SFR', 'SGF', 'SGH', 'SGM', 'SGP', 'SGR',
    #            'SHL', 'SHV', 'SIG', 'SIQ', 'SIV', 'SKC', 'SKI', 'SKT', 'SLC', 'SLK', 'SLR', 'SOL', 'SOM', 'SPK', 'SPL',
    #            'SRV', 'SRX', 'SSM', 'SST', 'STO', 'SUL', 'SUN', 'SVW', 'SWM', 'SXE', 'SXL', 'SXY', 'SYD', 'SYR', 'TAH',
    #            'TAW', 'TBR', 'TCL', 'TGP', 'TGR', 'TLS', 'TME', 'TNE', 'TOX', 'TPE', 'TPM', 'TRT', 'TTM', 'TWE', 'TZN',
    #            'UOS', 'UPD', 'URF', 'VAH', 'VCX', 'VLA', 'VLW', 'VOC', 'VRL', 'VRT', 'VTG', 'VVR', 'WAF', 'WBA', 'WBC',
    #            'WEB', 'WES', 'WFD', 'WGN', 'WGX', 'WHC', 'WLL', 'WOR', 'WOW', 'WPL', 'WPP', 'WSA', 'WTC', 'XAM', 'XRO',
    #            'YAL', 'YOJ', 'Z1P', 'ZEL', 'ZIM']
    active_short = 0
    first = timer()
    historical_data = import_data(csv_file, hdf_file, path)
    #tickers = historical_data["ticker"].unique().tolist()
    load = timer()
    print("import time: " + '{0:0.1f} seconds'.format(load - first))

    for ticker in tickers:
        mask = np.in1d(historical_data['ticker'].values, [ticker])
        historical_data_trim = historical_data[mask]
        historical_data_trim = tech_indicators(historical_data_trim)
        historical_data_trim = buy_and_sell_signals(historical_data_trim)
        historical_data_trim, buy_transactions, sell_transactions = trade(historical_data_trim, active_short)
        transactions = len(buy_transactions) + len(sell_transactions)
        annualised_return, annualised_return_ref, start_price, start_price_ref, end_price, end_price_ref, \
            start_date, start_date_ref, end_date, end_date_ref = calculate_returns(historical_data_trim)
        # print_results(ticker, transactions, annualised_return, annualised_return_ref, start_price, start_price_ref,
        #     end_price, end_price_ref, start_date, start_date_ref, end_date, end_date_ref, first)
        #results_to_csv(path, ticker, annualised_return, transactions, start_price, start_price_ref, end_price,
        #                end_price_ref, annualised_return_ref, start_date, start_date_ref, end_date, end_date_ref)
        plot_price(historical_data_trim, buy_transactions, sell_transactions)

    complete = timer()
    print("Final time: " + '{0:0.1f} seconds'.format(complete - load))

main()
