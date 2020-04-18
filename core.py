import logging
import numpy as np
from timeit import default_timer as timer
import config
import utility
from Result import Result
from Strategy import Strategy


def main():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)
    first = timer()
    historical_data = utility.import_data(config.csv_file, config.hdf_file, config.path)
    load = timer()
    print("Data import time: " + '{0:0.1f} seconds'.format(load - first))

    # Create strategies for backtesting
    strategies = []
    for i in np.arange(config.required_profit[0], config.required_profit[1], config.required_profit[2]):
        for j in np.arange(config.required_pct_change_max[0], config.required_pct_change_max[1],
                           config.required_pct_change_max[2]):
            for k in np.arange(config.required_volume[0], config.required_volume[1], config.required_volume[2]):
                strategies.append(Strategy(i, 0, j, k))


    strategy_count = len(strategies)
    i = 1
    for ticker in config.tickers:
        mask = np.in1d(historical_data['ticker'].values, [ticker])
        historical_data_trim = historical_data[mask]
        if len(historical_data_trim) == 0:
            logging.warning("Ticker " + ticker + " not in historical data.")
            continue

        for strategy in strategies:
            print("Strategy(" + str(strategy.required_profit) + ", " + str(strategy.required_pct_change_min) + ", " +
                  str(strategy.required_pct_change_max) + ", " + str(strategy.required_volume) + ")" + " "
                  + str(i) + " of " + str(strategy_count) + " of total strategies")

            result = Result(ticker, strategy, historical_data_trim)
            if config.write_results:
                performance = utility.init_performance_table()
                utility.init_transactions_table(performance)
                utility.result_to_db(result)
            else:
                utility.plot_price(result.data, ticker)
            i = i + 1

    complete = timer()
    print("Runtime: " + '{0:0.1f} seconds'.format(complete - load))


main()
