"""
data_loader.py
Downloads Jeff Sackmann's ATP/WTA match data from GitHub.
Columns used: tourney_date, tourney_name, tourney_level, surface, best_of,
              winner_name, loser_name, score,
              w_svpt, w_1stIn, w_1stWon, w_2ndWon,
              l_svpt, l_1stIn, l_1stWon, l_2ndWon
"""

import os
import requests
import pandas as pd
from io import StringIO

ATP_BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
WTA_BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data")

SERVE_COLS = [
    "tourney_date", "tourney_name", "tourney_level", "surface", "best_of",
    "winner_name", "loser_name", "score",
    "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
]

# WTA tourney_level mapping to match ATP conventions
# G=Grand Slam, PM=Premier Mandatory (≈Masters), P=Premier, I=International, F=Finals
WTA_MAIN_LEVELS = ["G", "PM", "P", "I", "F"]


def download_year(year: int, current_year: int = None, tour: str = "atp") -> pd.DataFrame | None:
    if tour == "wta":
        base_url = WTA_BASE_URL
        filename = f"wta_matches_{year}.csv"
    else:
        base_url = ATP_BASE_URL
        filename = f"atp_matches_{year}.csv"

    url   = f"{base_url}/{filename}"
    cache = os.path.join(CACHE_DIR, f"{tour}_{year}.csv")

    if os.path.exists(cache) and year != current_year:
        return pd.read_csv(cache, low_memory=False)
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text), low_memory=False)
        keep = [c for c in SERVE_COLS if c in df.columns]
        df = df[keep].copy()
        df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["tourney_date", "winner_name", "loser_name"])
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_csv(cache, index=False)
        print(f"  Downloaded {year}: {len(df)} matches")
        return df
    except Exception as e:
        print(f"  Failed {year}: {e}")
        return None


def load_data(start_year: int = 2010, end_year: int = 2026, tour: str = "atp") -> pd.DataFrame:
    import datetime
    current_year = datetime.date.today().year
    frames = []
    for year in range(start_year, end_year + 1):
        df = download_year(year, current_year=current_year, tour=tour)
        if df is not None:
            frames.append(df)
    if not frames:
        raise RuntimeError(f"No {tour.upper()} data downloaded.")
    combined = pd.concat(frames, ignore_index=True)
    combined["tourney_date"] = pd.to_datetime(combined["tourney_date"], errors="coerce")
    combined = combined.sort_values("tourney_date").reset_index(drop=True)
    if "tourney_level" in combined.columns:
        if tour == "wta":
            combined = combined[combined["tourney_level"].isin(WTA_MAIN_LEVELS)]
        else:
            combined = combined[combined["tourney_level"].isin(["G", "M", "A", "F"])]
    print(f"  Total matches loaded: {len(combined)}")
    return combined
