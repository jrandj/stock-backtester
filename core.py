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

    # Create strategies for backtesting
    strategies = []
    # for i in np.arange(1.1, 10, 0.1):
    #     for j in np.arange(0, 0.05, 0.01):
    #         for k in np.arange(3, 8, 0.1):
    #             strategies.append(Strategy(i, 0, j, k))

    strategies.append(Strategy(2.1, 0, 0, 7.7))

    i = 0
    length = len(strategies)
    for strategy in strategies:
        print("strategy: " + str(i) + " of total strategies: " + str(length))
        i = i + 1
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
