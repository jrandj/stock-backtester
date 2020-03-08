import logging
import numpy as np
from timeit import default_timer as timer
import config
import utility
from Result import Result
from Strategy import Strategy

def main():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)
    tickers = ["AJM"]
    # tickers = ["BRN", "NOR", "SAS", "KRR", "RAP", "ZYB", "UNL", "BOT", "SPT", "CGB", "MTC", "88E", "NHL", "SVA", "AC8",
    #            "AJM", "G88", "AVZ", "4CE", "MCT", "BAR", "IMU", "VOR", "BIT", "BD1", "CLA", "SCU"]

    first = timer()
    historical_data = utility.import_data(config.csv_file, config.hdf_file, config.path)
    load = timer()
    print("Data import time: " + '{0:0.1f} seconds'.format(load - first))

    # Create strategies for backtesting
    strategies = []
    # for i in np.arange(1.1, 10, 1):
    #     for j in np.arange(0, 0.05, 0.01):
    #         for k in np.arange(3, 8, 1):
    #             strategies.append(Strategy(i, 0, j, k))

    strategies.append(Strategy(3.1, 0, 0.04, 3))
    strategy_count = len(strategies)
    i = 1
    for ticker in tickers:
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
                # utility.results_to_csv(config.path, result)
                performance = utility.init_performance_table()
                utility.init_transactions_table(performance)
                utility.result_to_db(result)
            else:
                utility.plot_price(result.data, ticker)
            i = i + 1

    complete = timer()
    print("Runtime: " + '{0:0.1f} seconds'.format(complete - load))


main()
