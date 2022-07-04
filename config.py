import urllib

# Input
csv_file = r"\data.csv"
path = r"D:\Backups\Trading Project\Historical Data\ASX\Equities"
hdf_file = r"\data.h5"

# Strategy
write_results = 0
tickers = ["AJM"]
required_profit = [1.1, 5, 1]
required_pct_change_max = [0, 0.05, 0.01]
required_volume = [3, 8, 1]

# Database
params = urllib.parse.quote_plus("DRIVER={SQL Server Native Client 11.0};"
                                 "SERVER=DESKTOP-A4KARRF\MSSQLSERVER01;"
                                 "DATABASE=Analysis;"
                                 "Trusted_Connection=yes")
performance_table = "performance"
transactions_table = "transactions"
signals_table = "signals"

# Parameters
transaction_fee = 0.011
cash = 1
buy_and_hold_cash = 1