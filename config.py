import urllib

# Application config
csv_file = r"\data.csv"
path = r"C:\Users\James\Desktop\Backups\Trading Project\Historical Data\ASX\Equities"
hdf_file = r"\data.h5"
write_results = 0

# Database config
params = urllib.parse.quote_plus("DRIVER={SQL Server Native Client 11.0};"
                                 "SERVER=DESKTOP-A4KARRF\SQLEXPRESS;"
                                 "DATABASE=Analysis;"
                                 "Trusted_Connection=yes")
performance_table = "performance"
transactions_table = "transactions"
signals_table = "signals"

# Simulation config
transaction_fee = 0.011
cash = 1
buy_and_hold_cash = 1
