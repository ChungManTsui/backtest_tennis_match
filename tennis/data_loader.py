"""
data_loader.py
Downloads Jeff Sackmann's ATP match data from GitHub.
Columns used: tourney_date, tourney_name, tourney_level, surface, best_of,
              winner_name, loser_name, score,
              w_svpt, w_1stIn, w_1stWon, w_2ndWon,
              l_svpt, l_1stIn, l_1stWon, l_2ndWon
"""

import os
import requests
import pandas as pd
from io import StringIO

BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
CACHE_DIR = "tennis/data"

SERVE_COLS = [
    "tourney_date", "tourney_name", "tourney_level", "surface", "best_of",
    "winner_name", "loser_name", "score",
    "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
]


def download_year(year: int) -> pd.DataFrame | None:
    url = f"{BASE_URL}/atp_matches_{year}.csv"
    cache = os.path.join(CACHE_DIR, f"atp_{year}.csv")
    if os.path.exists(cache):
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


def load_data(start_year: int = 2010, end_year: int = 2024) -> pd.DataFrame:
    frames = []
    for year in range(start_year, end_year + 1):
        df = download_year(year)
        if df is not None:
            frames.append(df)
    if not frames:
        raise RuntimeError("No ATP data downloaded.")
    combined = pd.concat(frames, ignore_index=True)
    combined["tourney_date"] = pd.to_datetime(combined["tourney_date"], errors="coerce")
    combined = combined.sort_values("tourney_date").reset_index(drop=True)
    # Keep only main tour (exclude Challengers/ITF for cleaner data)
    if "tourney_level" in combined.columns:
        combined = combined[combined["tourney_level"].isin(["G", "M", "A", "F"])]
    print(f"  Total matches loaded: {len(combined)}")
    return combined
