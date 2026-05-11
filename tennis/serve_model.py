import numpy as np
import pandas as pd
from collections import defaultdict
from markov import prob_hold_game


def compute_serve_hold_pct(row: pd.Series, role: str):
    prefix = "w_" if role == "winner" else "l_"
    try:
        svpt = float(row[f"{prefix}svpt"])
        first_in = float(row[f"{prefix}1stIn"])
        first_won = float(row[f"{prefix}1stWon"])
        second_won = float(row[f"{prefix}2ndWon"])
        if svpt <= 0:
            return None
        return (first_won + second_won) / svpt
    except (KeyError, ValueError, TypeError):
        return None


def build_serve_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Walk-forward: for each match, look up rolling 20-match serve hold %
    from data BEFORE this match, then update history.
    Adds columns: p_hold_winner, p_hold_loser (as game-hold probabilities).
    """
    df = df.copy().sort_values("tourney_date").reset_index(drop=True)

    # player -> surface -> list of serve_point_win_pct values
    history: dict = defaultdict(lambda: defaultdict(list))

    p_hold_winner = []
    p_hold_loser = []

    for _, row in df.iterrows():
        surface = str(row.get("surface", "Hard"))
        winner = row["winner_name"]
        loser = row["loser_name"]

        def get_rolling(player, surf, n=20):
            records = history[player][surf]
            if len(records) >= 5:
                recent = records[-n:]
            else:
                # fall back to all surfaces combined
                all_r = []
                for s_vals in history[player].values():
                    all_r.extend(s_vals)
                recent = all_r[-n:]
            if len(recent) < 3:
                return None
            p_serve_pt = np.mean(recent)
            return prob_hold_game(p_serve_pt)

        p_hold_winner.append(get_rolling(winner, surface))
        p_hold_loser.append(get_rolling(loser, surface))

        # Update history AFTER recording prediction
        hold_w = compute_serve_hold_pct(row, "winner")
        hold_l = compute_serve_hold_pct(row, "loser")
        if hold_w is not None and 0.3 < hold_w < 1.0:
            history[winner][surface].append(hold_w)
        if hold_l is not None and 0.3 < hold_l < 1.0:
            history[loser][surface].append(hold_l)

    df["p_hold_winner"] = p_hold_winner
    df["p_hold_loser"] = p_hold_loser
    return df
