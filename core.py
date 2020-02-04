import logging
import numpy as np
from timeit import default_timer as timer
import config
import utility
from Result import Result
from Strategy import Strategy


def main():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)
    tickers = ["NOR"]
    first = timer()
    historical_data = utility.import_data(config.csv_file, config.hdf_file, config.path)
    load = timer()
    print("Data import time: " + '{0:0.1f} seconds'.format(load - first))
    strategy = Strategy(1.5, 0, 0.05, 8)

    for ticker in tickers:
        mask = np.in1d(historical_data['ticker'].values, [ticker])
        historical_data_trim = historical_data[mask]
        if len(historical_data_trim) == 0:
            logging.warning("Ticker " + ticker + " not in historical data.")
            continue

        result = Result(ticker, strategy, historical_data_trim)

        if config.write_results:
            utility.results_to_csv(config.path, result)
        else:
            utility.plot_price(result.data, ticker)

    complete = timer()
    print("Runtime: " + '{0:0.1f} seconds'.format(complete - load))


main()
