import os
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def combine_all_datasets(df_clean):
    carbon = pd.read_csv(os.path.join(CURRENT_DIR,"carbon","carbon_cleaned.csv"))
    flow = pd.read_csv(os.path.join(CURRENT_DIR,"flow","flows_combined.csv"))
    gas = pd.read_csv(os.path.join(CURRENT_DIR,"gas","gas_cleaned.csv"))
    gen = pd.read_csv(os.path.join(CURRENT_DIR,"gen","generation_cleaned.csv"))
    load = pd.read_csv(os.path.join(CURRENT_DIR,"load","full_cleaned.csv"))
    weather = pd.read_csv(os.path.join(CURRENT_DIR,"weather","weather_cleaned.csv"))

    carbon["utc"] = pd.to_datetime(carbon["utc"])
    flow["utc"] = pd.to_datetime(flow["utc"])
    gas["utc"] = pd.to_datetime(gas["utc"])
    gen["utc"] = pd.to_datetime(gen["utc"])
    load["utc"] = pd.to_datetime(load["utc"])
    weather["utc"] = pd.to_datetime(weather["utc"])
    
    df = df_clean.merge(carbon, on="utc", how="inner")
    df = df.merge(flow, on="utc", how="inner")
    df = df.merge(gas, on="utc", how="inner")
    df = df.merge(gen, on="utc", how="inner")
    df = df.merge(load, on="utc", how="inner")
    df = df.merge(weather, on="utc", how="inner")
    df = df.drop(["utc","value"],axis=1)
    
    path = os.path.join(CURRENT_DIR,"with_everything.csv")
    df.to_csv(path, index=False)
    print(path)
    print(df.shape)
    return df