import numpy as np
import pandas as pd
from collections import defaultdict
from markov import games_distribution_fast

MIN_EDGE = 0.04
MAX_STAKE_PCT = 0.05
QUARTER_KELLY = 0.25
BK_VIG = 0.09  # 9% bookmaker margin (realistic Bet365 tennis totals)

LEVEL_MAP = {
    "G": "Grand Slam",
    "M": "Masters 1000",
    "A": "ATP 500/250",
    "F": "ATP Finals",
}


def parse_score_games(score: str):
    if not isinstance(score, str):
        return None
    total = 0
    try:
        for part in score.strip().split():
            part = part.split("(")[0]
            if "-" in part:
                a, b = part.split("-")[:2]
                total += int(a) + int(b)
        return total if total > 0 else None
    except Exception:
        return None


def backtest(df: pd.DataFrame, starting_bankroll: float = 1000.0, vig: float = BK_VIG) -> dict:
    """
    Walk-forward backtest.

    Edge source: our model uses player-specific rolling serve stats.
    Bookmaker is simulated as using the surface-average serve hold %
    (rolling 200-match window) — a naive population model.
    When our player-specific distribution differs enough from the
    naive distribution, we have edge.
    vig: bookmaker margin applied to both sides (e.g. 0.09 = 9%)
    """
    df = df.copy().dropna(subset=["p_hold_winner", "p_hold_loser", "score"]).reset_index(drop=True)
    df = df[df["p_hold_winner"].between(0.35, 0.95)]
    df = df[df["p_hold_loser"].between(0.35, 0.95)]

    # Build rolling surface-average p_hold (bookmaker's naive model)
    # Uses all matches seen so far on that surface
    from serve_model import compute_serve_hold_pct
    from markov import prob_hold_game

    surface_history: dict = defaultdict(list)  # surface -> list of serve_pt_pct
    surface_avg_hold: dict = {}  # index -> (surface, avg_p_hold)

    # Pre-compute surface averages walk-forward
    surf_avgs = []
    for idx, row in df.iterrows():
        surf = str(row.get("surface", "Hard"))
        hist = surface_history[surf]
        if len(hist) >= 20:
            avg_pt = np.mean(hist[-200:])
            avg_hold = prob_hold_game(avg_pt)
        else:
            avg_hold = None
        surf_avgs.append(avg_hold)

        # Update surface history with both players' serve stats
        for role in ("winner", "loser"):
            val = compute_serve_hold_pct(row, role)
            if val is not None and 0.3 < val < 1.0:
                surface_history[surf].append(val)

    df["surf_avg_hold"] = surf_avgs

    records = []

    for _, row in df.iterrows():
        actual_games = parse_score_games(row["score"])
        if actual_games is None or actual_games < 6:
            continue

        best_of = int(row.get("best_of", 3))
        if best_of not in (3, 5):
            best_of = 3

        p_w = float(row["p_hold_winner"])
        p_l = float(row["p_hold_loser"])
        surf_avg = row["surf_avg_hold"]

        if surf_avg is None or np.isnan(surf_avg):
            continue

        # Our model: player-specific distribution
        try:
            our_dist = games_distribution_fast(p_w, p_l, best_of)
        except Exception:
            continue
        if not our_dist:
            continue

        # Bookmaker's naive model: both players at surface average
        try:
            bk_dist = games_distribution_fast(surf_avg, surf_avg, best_of)
        except Exception:
            continue
        if not bk_dist:
            continue

        # Bookmaker sets line at their median
        bk_sorted = sorted(bk_dist.items())
        cumulative = 0.0
        bk_line = None
        for g, p in bk_sorted:
            cumulative += p
            if cumulative >= 0.5:
                bk_line = g
                break
        if bk_line is None:
            continue

        # Bookmaker's implied probs — vig applied to both sides
        bk_p_over_true  = sum(p for g, p in bk_dist.items() if g > bk_line)
        bk_p_under_true = sum(p for g, p in bk_dist.items() if g < bk_line)

        if bk_p_over_true < 0.01 or bk_p_under_true < 0.01:
            continue

        bk_imp_over  = bk_p_over_true  * (1 + vig)
        bk_imp_under = bk_p_under_true * (1 + vig)
        bk_odds_over  = 1.0 / bk_imp_over
        bk_odds_under = 1.0 / bk_imp_under

        # Our model's probability at the bookmaker's line
        our_p_over  = sum(p for g, p in our_dist.items() if g > bk_line)
        our_p_under = sum(p for g, p in our_dist.items() if g < bk_line)

        edge_over  = our_p_over  - bk_imp_over
        edge_under = our_p_under - bk_imp_under

        if edge_over > MIN_EDGE and edge_over >= edge_under:
            bet_side, bet_prob, bet_odds, edge = "Over",  our_p_over,  bk_odds_over,  edge_over
        elif edge_under > MIN_EDGE:
            bet_side, bet_prob, bet_odds, edge = "Under", our_p_under, bk_odds_under, edge_under
        else:
            continue

        b = bet_odds - 1
        kelly = (b * bet_prob - (1 - bet_prob)) / b if b > 0 else 0
        if kelly <= 0:
            continue

        won = (actual_games > bk_line) if bet_side == "Over" else (actual_games < bk_line)
        level_name = LEVEL_MAP.get(str(row.get("tourney_level", "A")), "Other")

        records.append({
            "date":         row["tourney_date"],
            "tournament":   row.get("tourney_name", ""),
            "level":        level_name,
            "surface":      row.get("surface", ""),
            "best_of":      best_of,
            "winner":       row["winner_name"],
            "loser":        row["loser_name"],
            "p_hold_w":     round(p_w, 4),
            "p_hold_l":     round(p_l, 4),
            "surf_avg":     round(float(surf_avg), 4),
            "line":         bk_line,
            "actual_games": actual_games,
            "bet_side":     bet_side,
            "model_prob":   round(bet_prob, 4),
            "bk_odds":      round(bet_odds, 3),
            "edge":         round(edge, 4),
            "kelly":        round(kelly, 4),
            "won":          won,
        })

    bets_df = pd.DataFrame(records)
    if bets_df.empty:
        return {"bets_df": bets_df, "summary": {}, "by_level": {}, "by_surface": {}}

    return {
        "bets_df":    bets_df,
        "summary":    _simulate_bankroll(bets_df, starting_bankroll),
        "by_level":   {lvl: _simulate_bankroll(bets_df[bets_df["level"] == lvl], starting_bankroll)
                       for lvl in bets_df["level"].unique()},
        "by_surface": {sur: _simulate_bankroll(bets_df[bets_df["surface"] == sur], starting_bankroll)
                       for sur in bets_df["surface"].unique()},
    }


def _simulate_bankroll(bets_df: pd.DataFrame, starting_bankroll: float) -> dict:
    bankroll = starting_bankroll
    peak = bankroll
    max_dd = 0.0

    for _, bet in bets_df.iterrows():
        stake = bankroll * min(bet["kelly"] * QUARTER_KELLY, MAX_STAKE_PCT)
        bankroll += stake * (bet["bk_odds"] - 1) if bet["won"] else -stake
        peak = max(peak, bankroll)
        dd = (peak - bankroll) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    total = len(bets_df)
    wins  = int(bets_df["won"].sum())
    pnl   = bankroll - starting_bankroll
    be    = 1.0 / bets_df["bk_odds"].mean() * 100 if total else 0

    return {
        "total_bets":     total,
        "wins":           wins,
        "win_rate":       round(wins / total * 100, 1) if total else 0,
        "breakeven":      round(be, 1),
        "final_bankroll": round(bankroll, 2),
        "pnl":            round(pnl, 2),
        "roi":            round(pnl / starting_bankroll * 100, 1),
        "max_drawdown":   round(max_dd * 100, 1),
    }
