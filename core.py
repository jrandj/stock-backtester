import logging
import numpy as np
from timeit import default_timer as timer
import config
import utility
from Result import Result


def main():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)
    tickers = ["NOR"]
    first = timer()
    historical_data = utility.import_data(config.csv_file, config.hdf_file, config.path)
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
            utility.plot_price(result.data, ticker)

    complete = timer()
    print("Runtime: " + '{0:0.1f} seconds'.format(complete - load))


main()
