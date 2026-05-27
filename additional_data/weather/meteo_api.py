import requests
import pandas as pd
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

SINGLE_LOCATIONS = {
    "LT": {"latitude": 55.7, "longitude": 21.1},
    "LV": {"latitude": 57.0, "longitude": 24.5},
    "EE": {"latitude": 58.9, "longitude": 23.5},
}

MULTI_LOCATIONS = {
    "FI": [
        {"latitude": 64.0, "longitude": 24.5},  
        {"latitude": 61.0, "longitude": 21.9},  
        {"latitude": 60.3, "longitude": 19.9}, 
    ],
    "SE": [
        {"latitude": 57.7, "longitude": 12.0},  
        {"latitude": 56.5, "longitude": 16.0}, 
        {"latitude": 63.0, "longitude": 14.0},  
    ],
    "NO": [
        {"latitude": 58.5, "longitude": 6.0},   
        {"latitude": 63.5, "longitude": 10.0}, 
        {"latitude": 69.0, "longitude": 16.0}, 
    ],
    "PL": [
        {"latitude": 54.5, "longitude": 18.5},  
        {"latitude": 52.0, "longitude": 19.0},  
        {"latitude": 51.0, "longitude": 17.0}, 
    ],
}

def fetch_single(lat, lon):
    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2022-12-01",
        "end_date": "2025-12-25",
        "hourly": "windspeed_100m,shortwave_radiation",
        "timezone": "UTC"
    }
    r = requests.get(url, params=params)
    data = r.json()
    return {
        "utc": pd.to_datetime(data["hourly"]["time"]),
        "wind": data["hourly"]["windspeed_100m"],
        "solar": data["hourly"]["shortwave_radiation"],
    }

def fetch_averaged(country, points):
    all_wind = []
    all_solar = []
    times = None

    for p in points:
        result = fetch_single(country, p["latitude"], p["longitude"])
        all_wind.append(result["wind"])
        all_solar.append(result["solar"])
        if times is None:
            times = result["utc"]

    df = pd.DataFrame({
        "utc": times,
        f"wind_{country}": pd.DataFrame(all_wind).mean(axis=0),
        f"solar_{country}": pd.DataFrame(all_solar).mean(axis=0),
    })
    return df

def fetch_weather_single_country(country, coords):
    result = fetch_single(country, coords["latitude"], coords["longitude"])
    return pd.DataFrame({
        "utc": result["utc"],
        f"wind_{country}": result["wind"],
        f"solar_{country}": result["solar"],
    })

def fetch_all():
    dfs = []

    for country, coords in SINGLE_LOCATIONS.items():
        df = fetch_weather_single_country(coords)
        dfs.append(df)

    for country, points in MULTI_LOCATIONS.items():
        df = fetch_averaged(country, points)
        dfs.append(df)

    result = dfs[0]
    for right in dfs[1:]:
        result = result.merge(right, on='utc', how='inner')

    result = result[
        (result['utc'] >= '2023-01-01 00:00:00') &
        (result['utc'] <= '2025-10-25 00:00:00')
    ]

    result.to_csv(os.path.join(CURRENT_DIR, "weather.csv"), index=False)
    return result