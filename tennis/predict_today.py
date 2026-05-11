"""
predict_today.py — Barnett-Clarke ATP Total Games Predictor
Uses The Odds API for live Pinnacle/best-available total games lines.

Get a free API key at: https://the-odds-api.com (500 requests/month free)
Set it as: ODDS_API_KEY=your_key  or pass via --api-key argument

Run:
    python tennis/predict_today.py --api-key YOUR_KEY
    python tennis/predict_today.py  (reads ODDS_API_KEY env var)
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

import requests
import numpy as np
import pandas as pd
from data_loader import load_data
from serve_model import build_serve_stats, compute_serve_hold_pct
from markov import games_distribution_fast, prob_hold_game
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
START_YEAR   = 2015
END_YEAR     = 2025
MIN_EDGE     = 0.04
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
BOOKMAKER    = "bet365"     # fallback to pinnacle if bet365 not present

# ATP sport keys on The Odds API
ATP_SPORTS = [
    "tennis_atp_french_open",
    "tennis_atp_us_open",
    "tennis_atp_wimbledon",
    "tennis_atp_australian_open_singles",
    "tennis_atp_madrid_masters",
    "tennis_atp_rome_masters",
    "tennis_atp_monte_carlo_masters",
    "tennis_atp_canadian_open",
    "tennis_atp_cincinnati_masters",
    "tennis_atp_shanghai_masters",
    "tennis_atp_paris_masters",
    "tennis_atp_miami_open",
    "tennis_atp_indian_wells_masters",
    "tennis_atp_barcelona_open",
    "tennis_atp_halle_open",
    "tennis_atp_queens_club",
    "tennis_atp_eastbourne",
    "tennis_atp_washington",
    "tennis_atp_hamburg",
    "tennis_atp_gstaad",
    "tennis_atp_umag",
    "tennis_atp_kitzbuhel",
    "tennis_atp_montreal",
    "tennis_atp_winston_salem",
    "tennis_atp_metz",
    "tennis_atp_chengdu",
    "tennis_atp_tokyo",
    "tennis_atp_beijing",
    "tennis_atp_stockholm",
    "tennis_atp_antwerp",
    "tennis_atp_basel",
    "tennis_atp_vienna",
    "tennis_atp_atp_finals",
]
# ─────────────────────────────────────────────────────────────────────────────


def fetch_active_atp_sports(api_key: str) -> list[str]:
    """Query the API for currently active sports, return ATP tennis keys only."""
    try:
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports",
            params={"apiKey": api_key},
            timeout=10,
        )
        if r.status_code == 401:
            print("  ERROR: Invalid API key.")
            sys.exit(1)
        r.raise_for_status()
        all_sports = r.json()
        active = [
            s["key"] for s in all_sports
            if s.get("active") and "tennis_atp" in s["key"]
        ]
        if active:
            print(f"  Active ATP markets: {', '.join(active)}")
        return active
    except Exception as e:
        print(f"  Could not fetch active sports ({e}), falling back to default list.")
        return ATP_SPORTS


def fetch_odds(api_key: str) -> list[dict]:
    """Fetch all ATP total games odds from The Odds API."""
    sports_to_check = fetch_active_atp_sports(api_key)
    if not sports_to_check:
        print("  No active ATP markets found.")
        return []

    all_events = []
    last_r = None
    for sport in sports_to_check:
        url = ODDS_API_URL.format(sport=sport)
        try:
            r = requests.get(url, params={
                "apiKey":     api_key,
                "regions":    "eu",
                "markets":    "totals",
                "oddsFormat": "decimal",
            }, timeout=10)
            last_r = r
            if r.status_code == 404:
                continue
            if r.status_code == 401:
                print("  ERROR: Invalid API key.")
                sys.exit(1)
            r.raise_for_status()
            data = r.json()
            if data:
                all_events.extend(data)
                print(f"  {sport}: {len(data)} match(es) with totals lines")
            else:
                print(f"  {sport}: no totals lines yet")
        except requests.exceptions.RequestException as e:
            print(f"  {sport}: request failed ({e})")
            continue

    remaining = last_r.headers.get("x-requests-remaining", "?") if last_r else "?"
    print(f"  API requests remaining this month: {remaining}")
    return all_events


def parse_odds_events(events: list[dict]) -> list[dict]:
    """Extract (home, away, line, over_odds, under_odds) from API response."""
    parsed = []
    for ev in events:
        home = ev.get("home_team", "")
        away = ev.get("away_team", "")
        bookmakers = ev.get("bookmakers", [])

        # Prefer Bet365, fall back to Pinnacle, then skip others
        bk = next((b for b in bookmakers if b["key"] == "bet365"), None)
        if bk is None:
            bk = next((b for b in bookmakers if b["key"] == "pinnacle"), None)
        if bk is None:
            continue

        for market in bk.get("markets", []):
            if market["key"] != "totals":
                continue
            outcomes = {o["name"]: o for o in market.get("outcomes", [])}
            if "Over" not in outcomes or "Under" not in outcomes:
                continue
            line      = float(outcomes["Over"]["point"])
            over_odds = float(outcomes["Over"]["price"])
            under_odds = float(outcomes["Under"]["price"])
            parsed.append({
                "home":       home,
                "away":       away,
                "line":       line,
                "over_odds":  over_odds,
                "under_odds": under_odds,
                "bookmaker":  bk["key"],
            })
    return parsed


def build_player_lookup(df: pd.DataFrame) -> dict:
    """
    Walk-forward: build final rolling serve hold % for every player
    using all historical data. Used to look up today's players.
    """
    history: dict = defaultdict(lambda: defaultdict(list))

    for _, row in df.sort_values("tourney_date").iterrows():
        surf = str(row.get("surface", "Hard"))
        for role, name_col in [("winner", "winner_name"), ("loser", "loser_name")]:
            val = compute_serve_hold_pct(row, role)
            if val is not None and 0.3 < val < 1.0:
                history[row[name_col]][surf].append(val)

    # For each player+surface, compute rolling 20-match average → p_hold
    lookup = {}  # (player_lower, surface) -> p_hold
    for player, surf_data in history.items():
        for surf, vals in surf_data.items():
            if len(vals) >= 3:
                avg_pt = np.mean(vals[-20:])
                lookup[(player.lower(), surf)] = prob_hold_game(avg_pt)
        # Cross-surface fallback
        all_vals = []
        for v in surf_data.values():
            all_vals.extend(v)
        if len(all_vals) >= 3:
            avg_pt = np.mean(all_vals[-20:])
            lookup[(player.lower(), "any")] = prob_hold_game(avg_pt)

    return lookup


def fuzzy_lookup(name: str, surface: str, lookup: dict) -> float | None:
    """Find p_hold for a player, trying surface-specific then cross-surface."""
    key_surf = (name.lower(), surface)
    key_any  = (name.lower(), "any")
    if key_surf in lookup:
        return lookup[key_surf]
    if key_any in lookup:
        return lookup[key_any]
    # Try partial name match (API names may differ slightly from Sackmann)
    for (pname, psurf), val in lookup.items():
        if psurf == "any":
            parts = name.lower().split()
            if any(p in pname for p in parts if len(p) > 3):
                return val
    return None


def predict_match(home: str, away: str, line: float,
                  over_odds: float, under_odds: float,
                  surface: str, best_of: int,
                  lookup: dict) -> dict | None:
    p_home = fuzzy_lookup(home, surface, lookup)
    p_away = fuzzy_lookup(away, surface, lookup)

    if p_home is None or p_away is None:
        return {"skip": True, "reason": f"No serve data for {'both' if p_home is None and p_away is None else home if p_home is None else away}"}

    p_home = min(max(p_home, 0.35), 0.95)
    p_away = min(max(p_away, 0.35), 0.95)

    try:
        dist = games_distribution_fast(p_home, p_away, best_of)
    except Exception as e:
        return {"skip": True, "reason": str(e)}

    if not dist:
        return {"skip": True, "reason": "Empty distribution"}

    our_p_over  = sum(p for g, p in dist.items() if g > line)
    our_p_under = sum(p for g, p in dist.items() if g < line)

    imp_over  = 1.0 / over_odds
    imp_under = 1.0 / under_odds

    edge_over  = our_p_over  - imp_over
    edge_under = our_p_under - imp_under

    mean_g = sum(g * p for g, p in dist.items())

    result = {
        "skip":       False,
        "p_hold_home": round(p_home, 4),
        "p_hold_away": round(p_away, 4),
        "mean_games":  round(mean_g, 1),
        "our_p_over":  round(our_p_over, 4),
        "our_p_under": round(our_p_under, 4),
        "edge_over":   round(edge_over, 4),
        "edge_under":  round(edge_under, 4),
        "bet":         None,
    }

    if edge_over > MIN_EDGE and edge_over >= edge_under:
        b = over_odds - 1
        kelly = (b * our_p_over - (1 - our_p_over)) / b if b > 0 else 0
        result["bet"] = {
            "side":       "Over",
            "line":       line,
            "odds":       over_odds,
            "model_prob": round(our_p_over, 4),
            "edge":       round(edge_over * 100, 1),
            "kelly":      round(kelly, 4),
        }
    elif edge_under > MIN_EDGE:
        b = under_odds - 1
        kelly = (b * our_p_under - (1 - our_p_under)) / b if b > 0 else 0
        result["bet"] = {
            "side":       "Under",
            "line":       line,
            "odds":       under_odds,
            "model_prob": round(our_p_under, 4),
            "edge":       round(edge_under * 100, 1),
            "kelly":      round(kelly, 4),
        }

    return result


LOG_FILE = "tennis/data/bet_log.csv"
LOG_COLS = ["date", "surface", "home", "away", "bet_side", "line",
            "odds", "edge", "model_prob", "stake", "result", "pnl"]


def log_bets(bets: list, surface: str, today):
    """Append today's bets to the log CSV. result column left blank — fill in after match."""
    if not bets:
        return

    os.makedirs("tennis/data", exist_ok=True)
    rows = []
    for home, away, bet, stake in bets:
        rows.append({
            "date":       str(today),
            "surface":    surface,
            "home":       home,
            "away":       away,
            "bet_side":   bet["side"],
            "line":       bet["line"],
            "odds":       bet["odds"],
            "edge":       bet["edge"],
            "model_prob": bet["model_prob"],
            "stake":      round(stake, 2) if stake else "",
            "result":     "",   # fill in W or L after match
            "pnl":        "",   # auto-calculated when result filled
        })

    new_df = pd.DataFrame(rows, columns=LOG_COLS)

    if os.path.exists(LOG_FILE):
        existing = pd.read_csv(LOG_FILE)
        # Skip duplicates (same date + match already logged)
        key = ["date", "home", "away"]
        existing_keys = set(zip(existing["date"], existing["home"], existing["away"]))
        new_df = new_df[~new_df.apply(
            lambda r: (r["date"], r["home"], r["away"]) in existing_keys, axis=1
        )]
        if new_df.empty:
            print(f"\n  Bets already logged for today.")
            return
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(LOG_FILE, index=False)
    print(f"\n  {len(new_df)} bet(s) logged to {LOG_FILE}")
    print(f"  Open the file, fill in 'W' or 'L' in the result column after each match.")
    print(f"  Run 'python tennis/predict_today.py --pnl' to see your running P&L.")


def print_pnl_summary():
    """Read the log file, calculate P&L from filled-in results."""
    if not os.path.exists(LOG_FILE):
        return

    df = pd.read_csv(LOG_FILE)

    # Auto-calculate pnl where result is filled but pnl is empty
    updated = False
    for idx, row in df.iterrows():
        if str(row["result"]).strip().upper() in ("W", "L") and str(row["pnl"]).strip() == "":
            stake = float(row["stake"]) if str(row["stake"]).strip() != "" else 0
            if row["result"].strip().upper() == "W":
                df.at[idx, "pnl"] = round(stake * (float(row["odds"]) - 1), 2)
            else:
                df.at[idx, "pnl"] = round(-stake, 2)
            updated = True

    if updated:
        df.to_csv(LOG_FILE, index=False)

    # Summary of settled bets
    settled = df[df["result"].astype(str).str.strip().str.upper().isin(["W", "L"])].copy()
    if settled.empty:
        return

    settled["pnl"] = pd.to_numeric(settled["pnl"], errors="coerce").fillna(0)
    settled["stake"] = pd.to_numeric(settled["stake"], errors="coerce").fillna(0)

    total_bets   = len(settled)
    wins         = int((settled["result"].str.upper() == "W").sum())
    total_staked = settled["stake"].sum()
    total_pnl    = settled["pnl"].sum()
    roi          = total_pnl / total_staked * 100 if total_staked > 0 else 0

    print(f"\n{'='*65}")
    print(f"  RUNNING P&L TRACKER")
    print(f"{'='*65}")
    print(f"  Settled bets : {total_bets}")
    print(f"  Won          : {wins} ({wins/total_bets*100:.1f}%)")
    print(f"  Total staked : £{total_staked:.2f}")
    print(f"  Total P&L    : £{total_pnl:+.2f}")
    print(f"  ROI          : {roi:+.1f}%")

    # Last 5 bets
    print(f"\n  Last {min(5, len(settled))} settled bets:")
    for _, r in settled.tail(5).iterrows():
        print(f"  {r['date']} | {r['home']} vs {r['away']} | "
              f"{r['bet_side']} {r['line']} @ {r['odds']} | "
              f"{r['result'].upper()} | £{float(r['pnl']):+.2f}")
    print(f"{'='*65}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.environ.get("ODDS_API_KEY", ""))
    parser.add_argument("--bankroll", type=float, default=None)
    parser.add_argument("--surface", default="Hard",
                        help="Surface for today's matches (Hard/Clay/Grass)")
    parser.add_argument("--best-of", type=int, default=3)
    parser.add_argument("--pnl", action="store_true",
                        help="Just show P&L summary without fetching new odds")
    args = parser.parse_args()

    # P&L only mode
    if args.pnl:
        print_pnl_summary()
        return

    if not args.api_key:
        print("ERROR: No API key. Get one free at https://the-odds-api.com")
        print("Then run: python tennis/predict_today.py --api-key YOUR_KEY")
        sys.exit(1)

    print("=" * 65)
    print("  BARNETT-CLARKE ATP TOTAL GAMES — TODAY'S PREDICTIONS")
    print("=" * 65)

    # Load historical data and build player lookup
    print("\nLoading historical ATP data...")
    df = load_data(START_YEAR, END_YEAR)
    print("Building player serve profiles...")
    lookup = build_player_lookup(df)
    print(f"  {len(lookup)} player-surface profiles loaded")

    # Get bankroll
    bankroll = args.bankroll
    if bankroll is None:
        try:
            bankroll = float(input("\nEnter your bankroll in £ (for Kelly sizing): ").strip())
        except ValueError:
            print("Invalid — skipping stake sizing.")

    # Fetch live odds
    print("\nFetching today's ATP odds from The Odds API...")
    events = fetch_odds(args.api_key)
    if not events:
        print("  No ATP matches found today with total games lines.")
        return

    matches = parse_odds_events(events)
    print(f"  {len(matches)} match(es) with total games lines found.\n")

    # Predict each match
    bets = []
    print("=" * 65)
    for m in matches:
        home, away = m["home"], m["away"]
        print(f"  {home} vs {away}")
        print(f"  Line: {m['line']} | Over {m['over_odds']} / Under {m['under_odds']} ({m['bookmaker']})")

        result = predict_match(
            home, away, m["line"], m["over_odds"], m["under_odds"],
            args.surface, args.best_of, lookup
        )

        if result is None or result.get("skip"):
            reason = result.get("reason", "unknown") if result else "unknown"
            print(f"  SKIP — {reason}\n")
            continue

        print(f"  Model: mean {result['mean_games']} games | "
              f"P(over)={result['our_p_over']*100:.1f}% | P(under)={result['our_p_under']*100:.1f}%")
        print(f"  Edge: over {result['edge_over']*100:+.1f}% | under {result['edge_under']*100:+.1f}%")

        if result["bet"]:
            bet = result["bet"]
            print(f"  ★ BET: {bet['side']} {bet['line']} @ {bet['odds']} "
                  f"(edge {bet['edge']}%, model {bet['model_prob']*100:.1f}%)")
            if bankroll:
                stake = bankroll * min(bet["kelly"] * 0.25, 0.05)
                print(f"    Stake (Quarter Kelly, 5% cap): £{stake:.2f}")
                bets.append((home, away, bet, stake))
            else:
                bets.append((home, away, bet, None))
        else:
            print(f"  No edge — skip")
        print()

    # Summary
    print("=" * 65)
    if bets:
        print(f"  TODAY'S BETS ({len(bets)})")
        print("=" * 65)
        total_stake = 0
        for home, away, bet, stake in bets:
            stake_str = f"£{stake:.2f}" if stake else "—"
            print(f"  • {home} vs {away}: {bet['side']} {bet['line']} @ {bet['odds']} "
                  f"| edge {bet['edge']}% | stake {stake_str}")
            if stake:
                total_stake += stake
        if bankroll:
            print(f"\n  Total stake: £{total_stake:.2f} ({total_stake/bankroll*100:.1f}% of bankroll)")
    else:
        print("  NO BETS QUALIFY TODAY")
        print("  No matches passed the 4% edge threshold.")
    print("=" * 65)

    # Auto-log predictions to CSV
    log_bets(bets, args.surface, today=pd.Timestamp.now().date())
    print_pnl_summary()


if __name__ == "__main__":
    main()
