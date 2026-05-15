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


def _model_median(dist: dict) -> float:
    sorted_items = sorted(dist.items())
    cumulative = 0.0
    for g, p in sorted_items:
        cumulative += p
        if cumulative >= 0.5:
            return float(g)
    return float(sorted_items[-1][0])


def backtest(df: pd.DataFrame, starting_bankroll: float = 1000.0, vig: float = BK_VIG,
             bet_filter: str = "both", use_real_odds: bool = False) -> dict:
    """
    bet_filter: "both" | "over" | "under" | "under_opt"
    use_real_odds: if True and df has bf_line/bf_over_odds/bf_under_odds columns,
                   use real Betfair closing prices instead of simulated bookmaker.
    """
    """
    Walk-forward backtest.

    Your model: rolling 20-match player-specific serve stats (sharp).
    Simulated bookmaker: rolling 50-match player-specific serve stats (less sharp).
    Edge comes from your model being more responsive to recent form.
    Vig applied on top of bookmaker's line.
    """
    df = df.copy().dropna(subset=["p_hold_winner", "p_hold_loser", "score"]).reset_index(drop=True)
    df = df[df["p_hold_winner"].between(0.35, 0.95)]
    df = df[df["p_hold_loser"].between(0.35, 0.95)]

    from serve_model import compute_serve_hold_pct
    from markov import prob_hold_game

    # Bookmaker uses rolling 50-match window (less responsive than your 20-match model)
    bk_history: dict = defaultdict(lambda: defaultdict(list))
    bk_p_hold_winner = []
    bk_p_hold_loser  = []

    for _, row in df.iterrows():
        surface = str(row.get("surface", "Hard"))
        winner  = row["winner_name"]
        loser   = row["loser_name"]

        def get_bk_rolling(player, surf, n=50):
            records = bk_history[player][surf]
            if len(records) >= 5:
                recent = records[-n:]
            else:
                all_r = []
                for s_vals in bk_history[player].values():
                    all_r.extend(s_vals)
                recent = all_r[-n:]
            if len(recent) < 3:
                return None
            return prob_hold_game(np.mean(recent))

        bk_p_hold_winner.append(get_bk_rolling(winner, surface))
        bk_p_hold_loser.append(get_bk_rolling(loser, surface))

        for role, player in [("winner", winner), ("loser", loser)]:
            val = compute_serve_hold_pct(row, role)
            if val is not None and 0.3 < val < 1.0:
                bk_history[player][surface].append(val)

    df["bk_p_hold_winner"] = bk_p_hold_winner
    df["bk_p_hold_loser"]  = bk_p_hold_loser

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

        bk_w = row.get("bk_p_hold_winner")
        bk_l = row.get("bk_p_hold_loser")
        if bk_w is None or bk_l is None or np.isnan(float(bk_w)) or np.isnan(float(bk_l)):
            continue

        try:
            our_dist = games_distribution_fast(p_w, p_l, best_of)
        except Exception:
            continue
        if not our_dist:
            continue

        # Check if real Betfair odds are available for this row
        has_real_odds = (
            use_real_odds
            and "bf_line" in row.index
            and not pd.isna(row.get("bf_line"))
            and not pd.isna(row.get("bf_over_odds"))
            and not pd.isna(row.get("bf_under_odds"))
        )

        if has_real_odds:
            bk_line       = float(row["bf_line"])
            bk_odds_over  = float(row["bf_over_odds"])
            bk_odds_under = float(row["bf_under_odds"])
            # Remove Betfair commission (~5%) to get fair implied probabilities
            raw_sum       = (1.0 / bk_odds_over) + (1.0 / bk_odds_under)
            bk_imp_over   = (1.0 / bk_odds_over)  / raw_sum
            bk_imp_under  = (1.0 / bk_odds_under) / raw_sum
        else:
            try:
                bk_dist = games_distribution_fast(float(bk_w), float(bk_l), best_of)
            except Exception:
                continue
            if not bk_dist:
                continue

            # Bookmaker sets line at their median
            bk_line = _model_median(bk_dist)

            bk_p_over  = sum(p for g, p in bk_dist.items() if g > bk_line)
            bk_p_under = sum(p for g, p in bk_dist.items() if g < bk_line)

            if bk_p_over < 0.01 or bk_p_under < 0.01:
                continue

            # Bookmaker applies vig
            bk_imp_over  = bk_p_over  * (1 + vig)
            bk_imp_under = bk_p_under * (1 + vig)
            bk_odds_over  = 1.0 / bk_imp_over
            bk_odds_under = 1.0 / bk_imp_under

        # Your model's probability at the bookmaker's line
        our_p_over  = sum(p for g, p in our_dist.items() if g > bk_line)
        our_p_under = sum(p for g, p in our_dist.items() if g < bk_line)

        edge_over  = our_p_over  - bk_imp_over
        edge_under = our_p_under - bk_imp_under

        if edge_over > MIN_EDGE and edge_over >= edge_under:
            candidate_side, candidate_prob, candidate_odds, candidate_edge = "Over",  our_p_over,  bk_odds_over,  edge_over
        elif edge_under > MIN_EDGE:
            candidate_side, candidate_prob, candidate_odds, candidate_edge = "Under", our_p_under, bk_odds_under, edge_under
        else:
            continue

        # Apply bet_filter
        if bet_filter == "over"  and candidate_side != "Over":
            continue
        if bet_filter == "under" and candidate_side != "Under":
            continue

        # Under-optimised filter: bo5 + winner serves better than loser
        hold_gap = p_w - p_l
        if bet_filter == "under_opt":
            if candidate_side != "Under":
                continue
            if best_of != 5:
                continue
            if hold_gap <= 0:
                continue

        bet_side, bet_prob, bet_odds, edge = candidate_side, candidate_prob, candidate_odds, candidate_edge

        b = bet_odds - 1
        kelly = (b * bet_prob - (1 - bet_prob)) / b if b > 0 else 0
        if kelly <= 0:
            continue

        won = (actual_games > bk_line) if bet_side == "Over" else (actual_games < bk_line)
        level_name = LEVEL_MAP.get(str(row.get("tourney_level", "A")), "Other")
        hold_gap = p_w - p_l

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
            "hold_gap":     round(hold_gap, 4),
            "bk_p_hold_w":  round(float(bk_w), 4),
            "bk_p_hold_l":  round(float(bk_l), 4),
            "line":         bk_line,
            "actual_games": actual_games,
            "bet_side":     bet_side,
            "model_prob":   round(bet_prob, 4),
            "bk_odds":      round(bet_odds, 3),
            "edge":         round(edge, 4),
            "kelly":        round(kelly, 4),
            "won":          won,
            "real_odds":    has_real_odds,
        })

    bets_df = pd.DataFrame(records)
    if bets_df.empty:
        return {"bets_df": bets_df, "summary": {}, "by_level": {}, "by_surface": {}, "calibration": []}

    return {
        "bets_df":     bets_df,
        "summary":     _simulate_bankroll(bets_df, starting_bankroll),
        "by_level":    {lvl: _simulate_bankroll(bets_df[bets_df["level"] == lvl], starting_bankroll)
                        for lvl in bets_df["level"].unique()},
        "by_surface":  {sur: _simulate_bankroll(bets_df[bets_df["surface"] == sur], starting_bankroll)
                        for sur in bets_df["surface"].unique()},
        "calibration": _calibration(bets_df),
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


def _calibration(bets_df: pd.DataFrame) -> list:
    bins = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 1.00)]
    rows = []
    for lo, hi in bins:
        mask = bets_df["model_prob"].between(lo, hi)
        subset = bets_df[mask]
        if len(subset) == 0:
            continue
        actual_wr = subset["won"].mean() * 100
        pred_wr   = subset["model_prob"].mean() * 100
        rows.append({
            "prob_bucket":  f"{lo:.0%}-{hi:.0%}",
            "bets":         len(subset),
            "predicted_%":  round(pred_wr, 1),
            "actual_%":     round(actual_wr, 1),
            "diff":         round(actual_wr - pred_wr, 1),
        })
    return rows
