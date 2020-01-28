import logging
import os
import glob
import pandas as pd
import numpy as np
from timeit import default_timer as timer
import config
import utility
from Result import Result


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


def main():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)
    tickers = ["NOR"]
    first = timer()
    historical_data = import_data(config.csv_file, config.hdf_file, config.path)
    load = timer()
    print("Data import time: " + '{0:0.1f} seconds'.format(load - first))

    for ticker in tickers:
        mask = np.in1d(historical_data['ticker'].values, [ticker])
        historical_data_trim = historical_data[mask]
        if len(historical_data_trim) == 0:
            logging.warning("Ticker " + ticker + " not in historical data.")
            continue

        result = Result(ticker, historical_data_trim)
        if config.write_results:
            utility.results_to_csv(config.path, ticker, result.annualised_return, result.transactions,
                                   result.start_price,
                                   result.start_price_ref, result.end_price,
                                   result.end_price_ref, result.annualised_return_ref, result.start_date,
                                   result.start_date_ref,
                                   result.end_date, result.end_date_ref)
        else:
            utility.plot_price(result.data)

    complete = timer()
    print("Runtime: " + '{0:0.1f} seconds'.format(complete - load))


main()
