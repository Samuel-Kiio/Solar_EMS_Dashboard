# utils/prediction_pipeline.py

import pandas as pd
import requests
import joblib
from datetime import datetime, timedelta

NBO_TZ = "Africa/Nairobi"

def fetch_openmeteo_forecast(lat, lon):
    """
    Fetch next-day hourly forecast from Open-Meteo, already in Africa/Nairobi.
    We explicitly LOCALIZE to Africa/Nairobi (do NOT convert from UTC).
    """
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=global_tilted_irradiance,temperature_2m"
        f"&start_date={tomorrow}&end_date={tomorrow}"
        f"&timezone=Africa%2FNairobi"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    ts = pd.to_datetime(data["hourly"]["time"])      # tz-naive timestamps
    ts = ts.tz_localize(NBO_TZ)                      # localizing to Nairobi time. Key step. Do NOT convert from UTC.

    df = pd.DataFrame({
        "timestamp": ts,
        "Global Tilted Irradiation": data["hourly"]["global_tilted_irradiance"],
        "air_temp": data["hourly"]["temperature_2m"],
    })
    return df


def clean_forecast_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Small sanitization + features; keep timestamps tz-aware in Nairobi.
    """
    df = df.sort_values("timestamp").copy()

    # Smoothing any tiny spikes in GTI. Uses a 3-point rolling mean. Nice for how the visuals look. Not too important.
    df["Global Tilted Irradiation"] = (
        df["Global Tilted Irradiation"]
        .rolling(window=3, center=True, min_periods=1)
        .mean()
    )

    # Filling tiny gaps in temperature. Linear interpolating, then backfilling or frontfilling any edge NaNs.
    df["air_temp"] = df["air_temp"].interpolate().bfill().ffill()

    # Adding time-based features
    df["hour"] = df["timestamp"].dt.hour
    df["dayofyear"] = df["timestamp"].dt.dayofyear
    return df


def load_model_and_scaler():
    model = joblib.load("models/xgb_model.pkl")
    scaler = joblib.load("models/xgb_scaler.pkl")
    return model, scaler


def predict_next_day_production(lat, lon):
    """
    Return DataFrame with Nairobi-aware timestamps, the raw irradiance,
    and the predicted solar production (Wh per 30-min slot).
    """
    forecast = fetch_openmeteo_forecast(lat, lon)
    forecast = clean_forecast_data(forecast)

    model, scaler = load_model_and_scaler()

    feats = forecast[["Global Tilted Irradiation", "air_temp", "hour", "dayofyear"]]
    feats_scaled = scaler.transform(feats)

    forecast["predicted_solar_production"] = model.predict(feats_scaled)
    # Only returning key columns. Minor key step.
    return forecast[["timestamp", "Global Tilted Irradiation", "predicted_solar_production"]]
