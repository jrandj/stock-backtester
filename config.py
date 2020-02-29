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
table = "results"