"""
generate_sample_data.py
------------------------
Generates a realistic synthetic dataset mimicking the official Kaggle "US Accidents" schema.
Includes real metro cities (Atlanta, Dallas, Chicago, Los Angeles, Houston, Charlotte, Miami),
named streets, weather metrics, time stamps, and road infrastructure flags.

Schema is 100% compatible with the official Kaggle dataset:
https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

np.random.seed(42)

# City clusters with real coordinate centers and common major corridors/streets
METRO_CLUSTERS = [
    {
        "City": "Atlanta", "State": "GA", "County": "Fulton",
        "lat": 33.7490, "lng": -84.3880,
        "streets": [
            "I-85 N at Exit 87", "I-285 W Outer Loop", "I-75 S Downtown Connector",
            "Peachtree St NE", "Northside Dr NW", "Piedmont Rd NE", "Moreland Ave SE",
            "Buford Hwy NE", "MLK Jr Dr SW", "Cobb Pkwy SE"
        ]
    },
    {
        "City": "Dallas", "State": "TX", "County": "Dallas",
        "lat": 32.7767, "lng": -96.7970,
        "streets": [
            "I-35E S Stemmons Fwy", "US-75 N Central Expy", "I-30 E Tom Landry Fwy",
            "I-635 LBJ Fwy", "Loop 12 S", "Northwest Hwy", "Commerce St",
            "Greenville Ave", "Oak Lawn Ave", "Harry Hines Blvd"
        ]
    },
    {
        "City": "Chicago", "State": "IL", "County": "Cook",
        "lat": 41.8781, "lng": -87.6298,
        "streets": [
            "I-90 E Kennedy Expy", "I-94 S Dan Ryan Expy", "I-290 W Eisenhower Expy",
            "I-55 N Stevenson Expy", "DuSable Lake Shore Dr", "N Michigan Ave",
            "S Halsted St", "W Cicero Ave", "N Western Ave", "W 95th St"
        ]
    },
    {
        "City": "Los Angeles", "State": "CA", "County": "Los Angeles",
        "lat": 34.0522, "lng": -118.2437,
        "streets": [
            "I-405 N San Diego Fwy", "I-10 E Santa Monica Fwy", "US-101 S Hollywood Fwy",
            "I-5 S Golden State Fwy", "I-110 N Harbor Fwy", "Sepulveda Blvd",
            "Wilshire Blvd", "Sunset Blvd", "Ventura Blvd", "Western Ave"
        ]
    },
    {
        "City": "Houston", "State": "TX", "County": "Harris",
        "lat": 29.7604, "lng": -95.3698,
        "streets": [
            "I-45 N North Fwy", "I-610 Loop W", "I-10 E Katy Fwy", "US-59 S Southwest Fwy",
            "Beltway 8 Sam Houston Tollway", "Westheimer Rd", "Main St", "Richmond Ave",
            "Veterans Memorial Dr", "Gessner Rd"
        ]
    },
    {
        "City": "Charlotte", "State": "NC", "County": "Mecklenburg",
        "lat": 35.2271, "lng": -80.8431,
        "streets": [
            "I-77 S Bill Lee Fwy", "I-85 N Exit 38", "I-485 Outer Loop",
            "Independence Blvd", "South Blvd", "N Tryon St", "E Morehead St",
            "Providence Rd", "University City Blvd", "Monroe Rd"
        ]
    },
    {
        "City": "Miami", "State": "FL", "County": "Miami-Dade",
        "lat": 25.7617, "lng": -80.1918,
        "streets": [
            "I-95 S Express Lanes", "SR-836 W Dolphin Expy", "SR-826 Palmetto Expy",
            "Biscayne Blvd", "Brickell Ave", "SW 8th St (Calle Ocho)",
            "NW 27th Ave", "Collins Ave", "MacArthur Cswy", "Coral Way"
        ]
    }
]

WEATHER_CONDITIONS = ["Clear", "Fair", "Rain", "Light Rain", "Heavy Rain", "Snow", "Fog", "Thunderstorm", "Haze", "Cloudy", "Overcast"]
WEATHER_WEIGHTS = [0.35, 0.15, 0.12, 0.10, 0.04, 0.04, 0.05, 0.03, 0.02, 0.05, 0.05]


def generate_accidents(n_rows: int = 30000) -> pd.DataFrame:
    """Generates synthetic accident data correlated with realistic risk factors."""
    rows_per_metro = n_rows // len(METRO_CLUSTERS)
    records = []

    acc_id = 1
    for metro in METRO_CLUSTERS:
        for _ in range(rows_per_metro):
            # Select random street and add slight jitter around the metro center
            street = np.random.choice(metro["streets"])
            is_highway = street.startswith("I-") or street.startswith("US-") or "Fwy" in street or "Expy" in street

            # Location jitter (~0.05 deg spread, highway points cluster along line)
            jitter_lat = np.random.normal(0, 0.04)
            jitter_lng = np.random.normal(0, 0.04)
            start_lat = metro["lat"] + jitter_lat
            start_lng = metro["lng"] + jitter_lng

            # Date and time
            year = np.random.choice([2021, 2022, 2023], p=[0.25, 0.35, 0.40])
            month = np.random.randint(1, 13)
            day = np.random.randint(1, 29)
            hour_probs = np.array([
                0.015, 0.010, 0.010, 0.015, 0.025, 0.045, 0.075, 0.090, 0.080, 0.050,
                0.040, 0.045, 0.050, 0.050, 0.055, 0.075, 0.090, 0.085, 0.060, 0.040,
                0.035, 0.025, 0.020, 0.010
            ])
            hour_probs = hour_probs / hour_probs.sum()
            hour = int(np.random.choice(range(24), p=hour_probs))
            minute = np.random.randint(0, 60)
            second = np.random.randint(0, 60)
            start_time = pd.Timestamp(year, month, day, hour, minute, second)

            # Weather
            weather_p = np.array(WEATHER_WEIGHTS) / np.sum(WEATHER_WEIGHTS)
            weather = np.random.choice(WEATHER_CONDITIONS, p=weather_p)
            is_bad_weather = weather in ["Rain", "Heavy Rain", "Snow", "Fog", "Thunderstorm"]

            temp = np.random.normal(65, 18)
            visibility = 2.0 if weather in ["Fog", "Heavy Rain"] else (
                np.random.uniform(3.0, 7.0) if is_bad_weather else np.random.uniform(8.0, 10.0)
            )
            wind_speed = np.random.exponential(12 if is_bad_weather else 6)
            precipitation = np.random.exponential(0.35) if is_bad_weather else 0.0

            # Infrastructure flags
            junction = bool(np.random.choice([True, False], p=[0.45 if is_highway else 0.20, 0.55 if is_highway else 0.80]))
            traffic_signal = bool(np.random.choice([True, False], p=[0.10 if is_highway else 0.55, 0.90 if is_highway else 0.45]))
            crossing = bool(np.random.choice([True, False], p=[0.02 if is_highway else 0.30, 0.98 if is_highway else 0.70]))
            stop = bool(np.random.choice([True, False], p=[0.01 if is_highway else 0.15, 0.99 if is_highway else 0.85]))
            bump = bool(np.random.choice([True, False], p=[0.02, 0.98]))
            give_way = bool(np.random.choice([True, False], p=[0.03, 0.97]))
            roundabout = bool(np.random.choice([True, False], p=[0.02, 0.98]))

            # Risk and severity formula (correlated with physical world, but with realistic noise)
            is_rush = hour in [7, 8, 9, 16, 17, 18]
            is_weekend = start_time.dayofweek >= 5

            hazard_score = (
                (1.5 if is_bad_weather else 0.0)
                + (1.2 if visibility < 4.0 else 0.0)
                + (1.0 if junction else 0.0)
                + (0.7 if is_rush else 0.0)
                + (0.6 if is_highway else 0.0)
                + (0.4 if traffic_signal else 0.0)
                + (0.5 if (hour in [23, 0, 1, 2, 3] and is_weekend) else 0.0)
                + np.random.normal(0, 0.9)
            )

            # Assign Severity (1 to 4)
            if hazard_score > 3.8:
                severity = 4
            elif hazard_score > 2.4:
                severity = 3
            elif hazard_score > 1.2:
                severity = 2
            else:
                severity = 1

            records.append({
                "ID": f"A-{acc_id}",
                "Severity": severity,
                "Start_Time": str(start_time),
                "End_Time": str(start_time + pd.Timedelta(minutes=np.random.randint(30, 180))),
                "Start_Lat": round(start_lat, 6),
                "Start_Lng": round(start_lng, 6),
                "End_Lat": round(start_lat + np.random.normal(0, 0.002), 6),
                "End_Lng": round(start_lng + np.random.normal(0, 0.002), 6),
                "Distance(mi)": round(np.random.exponential(0.6), 3),
                "Description": f"Accident on {street} near {metro['City']}.",
                "Street": street,
                "City": metro["City"],
                "County": metro["County"],
                "State": metro["State"],
                "Zipcode": "00000",
                "Country": "US",
                "Timezone": "US/Eastern",
                "Airport_Code": "KXYZ",
                "Weather_Timestamp": str(start_time),
                "Temperature(F)": round(temp, 1),
                "Wind_Chill(F)": round(temp - 3, 1),
                "Humidity(%)": np.random.randint(30, 95),
                "Pressure(in)": round(np.random.normal(29.92, 0.2), 2),
                "Visibility(mi)": round(visibility, 1),
                "Wind_Direction": "W",
                "Wind_Speed(mph)": round(wind_speed, 1),
                "Precipitation(in)": round(precipitation, 2),
                "Weather_Condition": weather,
                "Amenity": False,
                "Bump": bump,
                "Crossing": crossing,
                "Give_Way": give_way,
                "Junction": junction,
                "No_Exit": False,
                "Railway": False,
                "Roundabout": roundabout,
                "Station": False,
                "Stop": stop,
                "Traffic_Calming": False,
                "Traffic_Signal": traffic_signal,
                "Turning_Loop": False,
                "Sunrise_Sunset": "Day" if 6 <= hour <= 19 else "Night",
                "Civil_Twilight": "Day" if 6 <= hour <= 19 else "Night",
                "Nautical_Twilight": "Day" if 5 <= hour <= 20 else "Night",
                "Astronomical_Twilight": "Day" if 5 <= hour <= 20 else "Night",
            })
            acc_id += 1

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic US Accidents dataset matching Kaggle schema.")
    parser.add_argument("--output", type=str, default="data/raw/US_Accidents_sample.csv")
    parser.add_argument("--rows", type=int, default=28000)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.rows} realistic sample accident records across major cities...")
    sample_df = generate_accidents(args.rows)
    sample_df.to_csv(out_path, index=False)
    print(f"Dataset successfully created at: {out_path}")
    print(f"Cities included: {sample_df['City'].unique().tolist()}")
    print("Severity distribution:")
    print(sample_df["Severity"].value_counts().sort_index())
