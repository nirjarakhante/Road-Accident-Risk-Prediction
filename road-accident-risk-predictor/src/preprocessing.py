"""
preprocessing.py
----------------
Cleans raw accident data (either Kaggle US Accidents or synthetic sample),
extracts temporal and weather features, groups locations into spatial grid cells,
and creates a balanced, leak-free training dataset for risk prediction.

Preserves City, State, and Primary Road identifiers for intuitive map exploration and search.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "Start_Time", "Start_Lat", "Start_Lng", "Severity",
    "Temperature(F)", "Visibility(mi)", "Wind_Speed(mph)",
    "Precipitation(in)", "Weather_Condition",
    "Junction", "Crossing", "Traffic_Signal",
]

OPTIONAL_META_COLUMNS = ["City", "State", "Street", "County"]


def load_and_clean(input_path: Path) -> pd.DataFrame:
    """Loads CSV and applies standard missing value cleaning and type conversions."""
    print(f"Reading {input_path} ...")
    df = pd.read_csv(input_path, low_memory=False)

    # Check required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    # Retain optional meta columns if they exist in the raw file
    keep_cols = [c for c in REQUIRED_COLUMNS if c in df.columns]
    for meta in OPTIONAL_META_COLUMNS:
        if meta in df.columns:
            keep_cols.append(meta)

    df = df[keep_cols].copy()

    # Drop rows without critical coordinates or severity
    df = df.dropna(subset=["Start_Lat", "Start_Lng", "Start_Time", "Severity"])

    # Impute missing weather and numerical features
    df["Temperature(F)"] = df["Temperature(F)"].fillna(df["Temperature(F)"].median())
    df["Visibility(mi)"] = df["Visibility(mi)"].fillna(df["Visibility(mi)"].median())
    df["Wind_Speed(mph)"] = df["Wind_Speed(mph)"].fillna(df["Wind_Speed(mph)"].median())
    df["Precipitation(in)"] = df["Precipitation(in)"].fillna(0.0)
    df["Weather_Condition"] = df["Weather_Condition"].fillna("Clear")

    # Clean boolean road features
    for col in ["Junction", "Crossing", "Traffic_Signal"]:
        df[col] = df[col].fillna(False).astype(bool)

    # Clean meta columns
    if "City" in df.columns:
        df["City"] = df["City"].fillna("Metro Area")
    else:
        df["City"] = "Metro Area"

    if "State" in df.columns:
        df["State"] = df["State"].fillna("US")
    else:
        df["State"] = "US"

    if "Street" in df.columns:
        df["Street"] = df["Street"].fillna("Main Road")
    else:
        df["Street"] = "Main Road"

    # Datetime parse
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df = df.dropna(subset=["Start_Time"])

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts hour, day of week, rush hour flags, and weekend flags."""
    df["Hour"] = df["Start_Time"].dt.hour
    df["DayOfWeek"] = df["Start_Time"].dt.dayofweek
    df["Month"] = df["Start_Time"].dt.month
    df["IsRushHour"] = df["Hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    return df


def simplify_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Simplifies diverse weather descriptions into standard behavioral categories."""
    def bucket(w: str) -> str:
        w = str(w).lower()
        if "clear" in w or "fair" in w:
            return "Clear"
        if "rain" in w or "drizzle" in w:
            return "Rain"
        if "snow" in w or "sleet" in w or "ice" in w or "wintry" in w:
            return "Snow"
        if "fog" in w or "mist" in w or "haze" in w or "smoke" in w:
            return "Fog"
        if "thunder" in w or "storm" in w or "squall" in w:
            return "Storm"
        if "cloud" in w or "overcast" in w:
            return "Cloudy"
        return "Other"

    df["WeatherGroup"] = df["Weather_Condition"].apply(bucket)
    return df


def grid_coordinates(df: pd.DataFrame, grid_size: float = 0.05) -> pd.DataFrame:
    """Snaps coordinates to ~5km grid cells (0.05 deg ≈ 5km)."""
    df["GridLat"] = (df["Start_Lat"] / grid_size).round() * grid_size
    df["GridLng"] = (df["Start_Lng"] / grid_size).round() * grid_size
    df["GridID"] = df["GridLat"].round(4).astype(str) + "_" + df["GridLng"].round(4).astype(str)
    return df


def aggregate_grid_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates point-in-time crashes into per-road/grid safety profiles.
    Target variable (Risk_Level) is created from historical frequency & severity,
    while input features reflect road infrastructure, weather risk, and time patterns.
    """
    def most_frequent(series):
        mode_vals = series.mode()
        return mode_vals.iloc[0] if not mode_vals.empty else series.iloc[0]

    agg = df.groupby("GridID").agg(
        GridLat=("GridLat", "first"),
        GridLng=("GridLng", "first"),
        City=("City", most_frequent),
        State=("State", most_frequent),
        PrimaryRoad=("Street", most_frequent),
        AccidentCount=("Severity", "count"),
        AvgSeverity=("Severity", "mean"),
        MaxSeverity=("Severity", "max"),
        PctJunction=("Junction", "mean"),
        PctCrossing=("Crossing", "mean"),
        PctTrafficSignal=("Traffic_Signal", "mean"),
        PctRushHour=("IsRushHour", "mean"),
        PctWeekend=("IsWeekend", "mean"),
        AvgVisibility=("Visibility(mi)", "mean"),
        AvgWindSpeed=("Wind_Speed(mph)", "mean"),
        AvgPrecipitation=("Precipitation(in)", "mean"),
        PctBadWeather=("WeatherGroup", lambda x: x.isin(["Rain", "Snow", "Fog", "Storm"]).mean()),
    ).reset_index()

    # Formulate composite risk score: frequency + severity
    freq_min, freq_max = agg["AccidentCount"].min(), agg["AccidentCount"].max()
    freq_norm = (agg["AccidentCount"] - freq_min) / (freq_max - freq_min + 1e-9)

    sev_min, sev_max = agg["AvgSeverity"].min(), agg["AvgSeverity"].max()
    sev_norm = (agg["AvgSeverity"] - sev_min) / (sev_max - sev_min + 1e-9)

    agg["RiskScore"] = (0.60 * freq_norm) + (0.40 * sev_norm)

    # Quantile thresholds: 55% Low Risk, 30% Medium Warning, 15% High Risk (Red Alert)
    q_low = agg["RiskScore"].quantile(0.55)
    q_med = agg["RiskScore"].quantile(0.85)

    def assign_risk_label(score):
        if score < q_low:
            return "Low"
        elif score < q_med:
            return "Medium"
        else:
            return "High"

    agg["Risk_Level"] = agg["RiskScore"].apply(assign_risk_label)

    # Determine primary danger factor for quick display on maps & tooltips
    def get_danger_factor(row):
        factors = []
        if row["PctJunction"] > 0.35:
            factors.append("Complex Highway Junction")
        if row["PctTrafficSignal"] > 0.40:
            factors.append("Congested Signalized Intersection")
        if row["PctBadWeather"] > 0.25:
            factors.append("Adverse Weather & Wet Pavement")
        if row["PctRushHour"] > 0.35:
            factors.append("Severe Rush Hour Peak")
        if row["AvgVisibility"] < 6.0:
            factors.append("Frequent Low Visibility")
        if not factors:
            factors.append("Mixed Traffic Density")
        return ", ".join(factors[:2])

    agg["PrimaryDangerFactor"] = agg.apply(get_danger_factor, axis=1)

    return agg


def run(input_path: Path, output_path: Path, grid_size: float = 0.05):
    df = load_and_clean(input_path)
    print(f"Loaded {len(df):,} valid records.")

    df = add_time_features(df)
    df = simplify_weather(df)
    df = grid_coordinates(df, grid_size)

    print("Aggregating to road grid risk table...")
    grid_df = aggregate_grid_risk(df)
    print(f"Generated {len(grid_df):,} distinct road segment grid cells.")
    print("Risk Level Breakdown:")
    print(grid_df["Risk_Level"].value_counts())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid_df.to_csv(output_path, index=False)
    print(f"Saved processed grid dataset to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw/US_Accidents_sample.csv")
    parser.add_argument("--output", type=str, default="data/processed/grid_risk.csv")
    parser.add_argument("--grid_size", type=float, default=0.05, help="Grid cell size in degrees (~0.05 ≈ 5km)")
    args = parser.parse_args()

    run(Path(args.input), Path(args.output), args.grid_size)
