import pandas as pd
import requests
from datetime import datetime, timedelta
import os

def fetch_forecast(lat=-1.2921, lon=36.8219, save_path="data/tomorrow_irradiance.csv"):
    tomorrow = datetime.utcnow().date() + timedelta(days=1)
    start_date = tomorrow.isoformat()
    end_date = start_date

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=shortwave_radiation,direct_normal_irradiance,diffuse_radiation"
        f"&timezone=Africa%2FNairobi"
        f"&start_date={start_date}&end_date={end_date}"
    )

    print(f"Requesting data for {start_date}...")
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"API request failed: {response.status_code}, {response.text}")
    
    data = response.json()
    if "hourly" not in data:
        raise Exception("Hourly forecast missing from response.")

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={
        "shortwave_radiation": "GHI",
        "direct_normal_irradiance": "DNI",
        "diffuse_radiation": "DHI"
    })
    
    df.to_csv(save_path, index=False)
    print(f"Forecast saved to: {save_path}")
    return df

# Testing the block
if __name__ == "__main__":
    fetch_forecast()

