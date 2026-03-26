import pandas as pd
import os
import requests
from io import BytesIO
from zipfile import ZipFile

# --------------------------
# CONFIGURATION
# --------------------------
years = [2018, 2019, 2020]   # Years to download
pair = 'EURUSD'              # Forex pair
output_folder = 'eurusd_data'
os.makedirs(output_folder, exist_ok=True)

# --------------------------
# TRUEFX DOWNLOAD FUNCTION
# --------------------------
def download_truefx_year(year):
    url = f"http://www.truefx.com/dev/data/history/{year}/EURUSD_H1.zip"
    print(f"Downloading {url} ...")
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to download {year}")
        return None
    return BytesIO(response.content)

# --------------------------
# PROCESS EACH YEAR
# --------------------------
all_data = []

for year in years:
    zip_file = download_truefx_year(year)
    if zip_file is None:
        continue
    with ZipFile(zip_file) as z:
        # TrueFX has CSV files inside the zip
        for filename in z.namelist():
            if filename.endswith('.csv'):
                with z.open(filename) as f:
                    df = pd.read_csv(f, header=None, names=['Date','Time','Bid','Ask','Volume'])
                    # Convert to single OHLCV bar
                    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y%m%d %H%M%S')
                    df['Open'] = df['Bid']  # Using Bid as proxy
                    df['High'] = df['Bid']
                    df['Low'] = df['Bid']
                    df['Close'] = df['Bid']
                    df = df[['Datetime','Open','High','Low','Close','Volume']]
                    all_data.append(df)

# --------------------------
# CONCAT AND SAVE CSV
# --------------------------
if all_data:
    final_df = pd.concat(all_data)
    final_df = final_df.sort_values('Datetime').reset_index(drop=True)
    final_df.rename(columns={'Datetime':'Gmt time'}, inplace=True)
    final_df.to_csv('EURUSD_hourly_multi_year.csv', index=False)
    print("Saved CSV: EURUSD_hourly_multi_year.csv")
else:
    print("No data downloaded.")