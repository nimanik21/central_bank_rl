
import json
import requests
import pandas as pd
import os

def fetch_fred_data(series_id, api_key, start=None, end=None):
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json"
    if start and end:
        url += f"&observation_start={start}&observation_end={end}"
    data = requests.get(url).json()
    
    if "observations" not in data:
        print(f"Error fetching {series_id}: Unexpected response format")
        return None
    
    df = pd.DataFrame(data["observations"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df.set_index("date", inplace=True)
    
    return df

def download_multiple_series(api_key, series_list, start=None, end=None, out_dir="data/raw"):
    """
    Download multiple series from FRED and save each as a CSV in data/raw/.
    """
    os.makedirs(out_dir, exist_ok=True)

    for series_id in series_list:
        df = fetch_fred_data(series_id, api_key, start, end)
        if df is not None:
            csv_path = os.path.join(out_dir, f"{series_id}.csv")
            df.to_csv(csv_path)
            print(f"Saved {series_id} to {csv_path}")
        else:
            print(f"Failed to fetch {series_id}")

if __name__ == "__main__":
    # EXAMPLE USAGE:
    # For security, store your actual API key in a JSON file or an environment variable
    # We'll assume you have a JSON with your key at the path below:
    
    with open("/content/drive/MyDrive/API/api_keys.json", "r") as f:
        fred_api_key = json.load(f)["FRED_API_KEY"]
    
    series_list = ["CPIAUCSL", "UNRATE", "FEDFUNDS"]
    download_multiple_series(
        api_key=fred_api_key, 
        series_list=series_list, 
        start="2000-01-01", 
        end="2025-01-01",
        out_dir="/content/drive/MyDrive/central_bank_rl/data/raw"
    )
