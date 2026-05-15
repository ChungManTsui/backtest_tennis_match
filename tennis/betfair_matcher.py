"""
betfair_matcher.py
Match Betfair TOTAL_GAMES odds to Sackmann match data.
Uses fuzzy name matching to handle name differences (e.g. "Stanislas Wawrinka" vs "Stan Wawrinka").
"""
import pandas as pd
import numpy as np
import os
import re
from difflib import SequenceMatcher


def normalize_name(name: str) -> str:
    """Lowercase, remove punctuation, normalize spaces."""
    name = name.lower().strip()
    name = re.sub(r"['\-\.]", " ", name)
    name = re.sub(r"\s+", " ", name)
    return name


def name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_name(a), normalize_name(b)).ratio()


def best_match(name: str, candidates: list, threshold: float = 0.75) -> str | None:
    best_score = 0.0
    best_name = None
    for c in candidates:
        s = name_similarity(name, c)
        if s > best_score:
            best_score = s
            best_name = c
    return best_name if best_score >= threshold else None


def load_betfair_odds(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def match_odds_to_sackmann(sackmann_df: pd.DataFrame, betfair_df: pd.DataFrame,
                            date_window_days: int = 1) -> pd.DataFrame:
    """
    For each row in sackmann_df, find the matching Betfair TOTAL_GAMES record.
    Match on: date (±window), and both player names appear in the Betfair record.
    Returns sackmann_df with added columns: bf_line, bf_over_odds, bf_under_odds.
    """
    sackmann_df = sackmann_df.copy()
    sackmann_df['bf_line']       = np.nan
    sackmann_df['bf_over_odds']  = np.nan
    sackmann_df['bf_under_odds'] = np.nan

    betfair_df = betfair_df.copy()
    betfair_df['date'] = pd.to_datetime(betfair_df['date'])

    matched = 0
    for idx, row in sackmann_df.iterrows():
        match_date = pd.to_datetime(row['tourney_date'])
        winner = str(row['winner_name'])
        loser  = str(row['loser_name'])

        # Filter by date window
        mask = (betfair_df['date'] >= match_date - pd.Timedelta(days=date_window_days)) & \
               (betfair_df['date'] <= match_date + pd.Timedelta(days=date_window_days))
        candidates = betfair_df[mask]

        best_score = 0.0
        best_row = None
        for _, bf in candidates.iterrows():
            # Both players must match (either order)
            pa, pb = str(bf['player_a']), str(bf['player_b'])
            score_fwd = min(name_similarity(winner, pa), name_similarity(loser, pb))
            score_rev = min(name_similarity(winner, pb), name_similarity(loser, pa))
            score = max(score_fwd, score_rev)
            if score > best_score:
                best_score = score
                best_row = bf

        if best_row is not None and best_score >= 0.75:
            sackmann_df.at[idx, 'bf_line']       = best_row['line']
            sackmann_df.at[idx, 'bf_over_odds']  = best_row['over_odds']
            sackmann_df.at[idx, 'bf_under_odds'] = best_row['under_odds']
            matched += 1

    total = len(sackmann_df)
    print(f'  Matched {matched}/{total} ({matched/total*100:.1f}%) Sackmann rows to Betfair odds')
    return sackmann_df
