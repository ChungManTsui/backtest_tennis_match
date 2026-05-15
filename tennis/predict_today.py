"""
predict_today.py — Barnett-Clarke Tennis Total Games Predictor
Supports ATP and WTA, with bet filter flags.

Run:
    python tennis/predict_today.py --atp --filter both --api-key YOUR_KEY
    python tennis/predict_today.py --wta --filter under --api-key YOUR_KEY
    python tennis/predict_today.py --atp --wta --filter both --api-key YOUR_KEY
    python tennis/predict_today.py --atp --pnl
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
from datetime import date
from collections import defaultdict

from data_loader import load_data
from serve_model import build_serve_stats, compute_serve_hold_pct
from markov import games_distribution_fast, prob_hold_game

# ── Config ─────────────────────────────────────────────────────────────────────
START_YEAR = 2015
END_YEAR   = 2025
MIN_EDGE   = 0.04
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"

GRAND_SLAM_KEYS = {"french_open", "wimbledon", "us_open", "australian_open"}

LOG_COLS = ["date", "tour", "surface", "home", "away", "best_of",
            "bet_side", "line", "odds", "edge", "model_prob", "kelly",
            "stake", "result", "pnl"]
# ───────────────────────────────────────────────────────────────────────────────


def _log_file(tour: str) -> str:
    return f"tennis/data/bet_log_{tour}.csv"


def fetch_active_sports(api_key: str, tour: str) -> list[str]:
    prefix = f"tennis_{tour}"
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
        active = [s["key"] for s in r.json() if s.get("active") and s["key"].startswith(prefix)]
        if active:
            print(f"  Active {tour.upper()} markets: {', '.join(active)}")
        else:
            print(f"  No active {tour.upper()} markets found.")
        return active
    except Exception as e:
        print(f"  Could not fetch active sports ({e})")
        return []


def is_grand_slam(sport_key: str) -> bool:
    return any(gs in sport_key for gs in GRAND_SLAM_KEYS)


def fetch_odds(api_key: str, tour: str) -> list[dict]:
    """Fetch all total-games odds for a tour. Returns list of parsed match dicts."""
    sports = fetch_active_sports(api_key, tour)
    if not sports:
        return []

    import zoneinfo
    uk_tz   = zoneinfo.ZoneInfo("Europe/London")
    today_uk = date.today()

    all_matches = []
    last_r = None

    for sport in sports:
        best_of = 5 if is_grand_slam(sport) else 3
        try:
            r = requests.get(ODDS_API_URL.format(sport=sport), params={
                "apiKey":     api_key,
                "regions":    "uk,eu",
                "markets":    "totals",
                "oddsFormat": "decimal",
            }, timeout=10)
            last_r = r
            if r.status_code in (404, 422):
                continue
            if r.status_code == 401:
                print("  ERROR: Invalid API key.")
                sys.exit(1)
            r.raise_for_status()
            events = r.json()
            if not events:
                continue

            print(f"  {sport}: {len(events)} event(s)")
            for ev in events:
                commence = ev.get("commence_time", "")
                try:
                    uk_dt = pd.Timestamp(commence, tz="UTC").tz_convert(uk_tz)
                    if uk_dt.date() != today_uk:
                        continue
                    time_uk = uk_dt.strftime("%H:%M")
                except Exception:
                    time_uk = "?"

                home = ev.get("home_team", "")
                away = ev.get("away_team", "")

                # Collect all bookmaker prices per line
                line_prices: dict[float, dict[str, list[float]]] = {}
                for bk in ev.get("bookmakers", []):
                    for market in bk.get("markets", []):
                        if market["key"] != "totals":
                            continue
                        for outcome in market.get("outcomes", []):
                            ln   = float(outcome["point"])
                            side = outcome["name"]
                            line_prices.setdefault(ln, {"Over": [], "Under": []})
                            line_prices[ln][side].append(float(outcome["price"]))

                if not line_prices:
                    continue

                best_line = max(line_prices, key=lambda l: len(line_prices[l]["Over"]) + len(line_prices[l]["Under"]))
                prices = line_prices[best_line]
                if not prices["Over"] or not prices["Under"]:
                    continue

                all_matches.append({
                    "home":       home,
                    "away":       away,
                    "line":       best_line,
                    "over_odds":  max(prices["Over"]),
                    "under_odds": max(prices["Under"]),
                    "avg_over":   sum(prices["Over"])  / len(prices["Over"]),
                    "avg_under":  sum(prices["Under"]) / len(prices["Under"]),
                    "best_of":    best_of,
                    "sport":      sport,
                    "time_uk":    time_uk,
                })
        except requests.exceptions.RequestException as e:
            print(f"  {sport}: request failed ({e})")
            continue

    remaining = last_r.headers.get("x-requests-remaining", "?") if last_r else "?"
    print(f"  API requests remaining: {remaining}")
    return all_matches


def build_lookup(df: pd.DataFrame) -> dict:
    history: dict = defaultdict(lambda: defaultdict(list))
    for _, row in df.sort_values("tourney_date").iterrows():
        surf = str(row.get("surface", "Hard"))
        for role, col in [("winner", "winner_name"), ("loser", "loser_name")]:
            val = compute_serve_hold_pct(row, role)
            if val is not None and 0.3 < val < 1.0:
                history[row[col]][surf].append(val)
    lookup = {}
    for player, surf_data in history.items():
        for surf, vals in surf_data.items():
            if len(vals) >= 3:
                lookup[(player.lower(), surf)] = prob_hold_game(np.mean(vals[-20:]))
        all_vals = [v for vs in surf_data.values() for v in vs]
        if len(all_vals) >= 3:
            lookup[(player.lower(), "any")] = prob_hold_game(np.mean(all_vals[-20:]))
    return lookup


def fuzzy_lookup(name: str, surface: str, lookup: dict):
    for key in [(name.lower(), surface), (name.lower(), "any")]:
        if key in lookup:
            return lookup[key]
    for (pname, psurf), val in lookup.items():
        if psurf == "any":
            if any(p in pname for p in name.lower().split() if len(p) > 3):
                return val
    return None


def predict_match(home, away, line, over_odds, under_odds, surface, best_of,
                  lookup, avg_over=None, avg_under=None):
    p_w = fuzzy_lookup(home, surface, lookup)
    p_l = fuzzy_lookup(away, surface, lookup)
    if p_w is None or p_l is None:
        missing = "both" if p_w is None and p_l is None else (home if p_w is None else away)
        return {"skip": True, "reason": f"No serve data for {missing}"}

    p_w = min(max(p_w, 0.35), 0.95)
    p_l = min(max(p_l, 0.35), 0.95)

    try:
        dist = games_distribution_fast(p_w, p_l, best_of)
    except Exception as e:
        return {"skip": True, "reason": str(e)}
    if not dist:
        return {"skip": True, "reason": "Empty distribution"}

    our_p_over  = sum(p for g, p in dist.items() if g > line)
    our_p_under = sum(p for g, p in dist.items() if g < line)
    ref_over    = avg_over  or over_odds
    ref_under   = avg_under or under_odds
    edge_over   = our_p_over  - 1.0 / ref_over
    edge_under  = our_p_under - 1.0 / ref_under
    mean_g      = sum(g * p for g, p in dist.items())

    result = {
        "skip": False,
        "p_hold_home": round(p_w, 4),
        "p_hold_away": round(p_l, 4),
        "mean_games":  round(mean_g, 1),
        "our_p_over":  round(our_p_over, 4),
        "our_p_under": round(our_p_under, 4),
        "edge_over":   round(edge_over, 4),
        "edge_under":  round(edge_under, 4),
        "bet": None,
    }

    if edge_over > MIN_EDGE and edge_over >= edge_under:
        b = over_odds - 1
        kelly = (b * our_p_over - (1 - our_p_over)) / b if b > 0 else 0
        result["bet"] = {"side": "Over", "line": line, "odds": over_odds,
                         "model_prob": round(our_p_over, 4),
                         "edge": round(edge_over * 100, 1),
                         "kelly": round(kelly, 4)}
    elif edge_under > MIN_EDGE:
        b = under_odds - 1
        kelly = (b * our_p_under - (1 - our_p_under)) / b if b > 0 else 0
        result["bet"] = {"side": "Under", "line": line, "odds": under_odds,
                         "model_prob": round(our_p_under, 4),
                         "edge": round(edge_under * 100, 1),
                         "kelly": round(kelly, 4)}
    return result


def apply_filter(bet: dict | None, bet_filter: str, best_of: int) -> bool:
    if bet is None:
        return False
    if bet_filter == "both":
        return True
    if bet_filter == "over":
        return bet["side"] == "Over"
    if bet_filter == "under":
        return bet["side"] == "Under"
    if bet_filter == "under_opt":
        return bet["side"] == "Under" and best_of == 5
    return False


def log_bets(bets: list, tour: str, surface: str, today: str):
    if not bets:
        return
    os.makedirs("tennis/data", exist_ok=True)
    log_file = _log_file(tour)
    rows = []
    for home, away, best_of, bet, stake in bets:
        rows.append({
            "date":       today,
            "tour":       tour.upper(),
            "surface":    surface,
            "home":       home,
            "away":       away,
            "best_of":    best_of,
            "bet_side":   bet["side"],
            "line":       bet["line"],
            "odds":       bet["odds"],
            "edge":       bet["edge"],
            "model_prob": bet["model_prob"],
            "kelly":      bet["kelly"],
            "stake":      round(stake, 2) if stake else "",
            "result":     "",
            "pnl":        "",
        })
    new_df = pd.DataFrame(rows, columns=LOG_COLS)
    if os.path.exists(log_file):
        existing = pd.read_csv(log_file)
        existing_keys = set(zip(existing["date"], existing["home"], existing["away"]))
        new_df = new_df[~new_df.apply(
            lambda r: (r["date"], r["home"], r["away"]) in existing_keys, axis=1)]
        if new_df.empty:
            print(f"  Bets already logged for today.")
            return
        pd.concat([existing, new_df], ignore_index=True).to_csv(log_file, index=False)
    else:
        new_df.to_csv(log_file, index=False)
    print(f"  {len(new_df)} bet(s) logged to {log_file}")


def print_pnl_summary(tour: str):
    log_file = _log_file(tour)
    if not os.path.exists(log_file):
        return
    df = pd.read_csv(log_file)
    settled = df[df["result"].astype(str).str.strip().str.upper().isin(["W", "L"])].copy()
    if settled.empty:
        return
    settled["pnl"]   = pd.to_numeric(settled["pnl"],   errors="coerce").fillna(0)
    settled["stake"] = pd.to_numeric(settled["stake"], errors="coerce").fillna(0)
    total_bets   = len(settled)
    wins         = int((settled["result"].str.upper() == "W").sum())
    total_staked = settled["stake"].sum()
    total_pnl    = settled["pnl"].sum()
    roi          = total_pnl / total_staked * 100 if total_staked > 0 else 0
    print(f"\n{'='*65}")
    print(f"  {tour.upper()} RUNNING P&L")
    print(f"{'='*65}")
    print(f"  Settled : {total_bets} | Won: {wins} ({wins/total_bets*100:.1f}%)")
    print(f"  Staked  : £{total_staked:.2f} | P&L: £{total_pnl:+.2f} | ROI: {roi:+.1f}%")
    for _, r in settled.tail(5).iterrows():
        pnl_val = float(r["pnl"]) if str(r["pnl"]).strip() != "" else 0
        print(f"  {r['date']} | {r['home']} vs {r['away']} | "
              f"{r['bet_side']} {r['line']} @ {r['odds']} | "
              f"{str(r['result']).upper()} | £{pnl_val:+.2f}")
    print(f"{'='*65}")


def run_tour(tour: str, api_key: str, surface: str, bet_filter: str,
             bankroll: float | None, kelly_fraction: float, max_stake: float,
             lookup: pd.DataFrame):
    today = str(date.today())
    print(f"\n{'='*65}")
    print(f"  {tour.upper()} — TODAY'S PREDICTIONS ({today})")
    print(f"  Filter: {bet_filter} | Surface: {surface}")
    print(f"{'='*65}")

    print(f"\nFetching {tour.upper()} odds...")
    matches = fetch_odds(api_key, tour)
    if not matches:
        print(f"  No {tour.upper()} matches with total games lines today.")
        return []

    print(f"  {len(matches)} match(es) found.\n")

    bets = []
    all_matches_info = []
    print(f"{'='*65}")
    for m in matches:
        home, away = m["home"], m["away"]
        best_of    = m["best_of"]
        print(f"  {home} vs {away}  [bo{best_of}, {m.get('time_uk','?')} UK]")
        print(f"  Line: {m['line']} | Over {m['over_odds']} / Under {m['under_odds']}")

        result = predict_match(
            home, away, m["line"], m["over_odds"], m["under_odds"],
            surface, best_of, lookup,
            avg_over=m.get("avg_over"), avg_under=m.get("avg_under"),
        )

        if result is None or result.get("skip"):
            reason = result.get("reason", "unknown") if result else "unknown"
            print(f"  SKIP — {reason}\n")
            all_matches_info.append((home, away, m["line"], m["over_odds"], m["under_odds"], None, None, m.get("time_uk","?")))
            continue

        print(f"  Model: mean {result['mean_games']} games | "
              f"P(over)={result['our_p_over']*100:.1f}% | P(under)={result['our_p_under']*100:.1f}%")
        print(f"  Edge:  over {result['edge_over']*100:+.1f}% | under {result['edge_under']*100:+.1f}%")

        bet = result["bet"]
        if bet and apply_filter(bet, bet_filter, best_of):
            stake = bankroll * min(bet["kelly"] * kelly_fraction, max_stake) if bankroll else None
            print(f"  ★ BET: {bet['side']} {bet['line']} @ {bet['odds']} "
                  f"(edge {bet['edge']}%, model {bet['model_prob']*100:.1f}%)")
            if stake:
                print(f"    Stake ({kelly_fraction*100:.0f}% Kelly, {max_stake*100:.0f}% cap): £{stake:.2f}")
            bets.append((home, away, best_of, bet, stake))
            all_matches_info.append((home, away, m["line"], m["over_odds"], m["under_odds"], bet, stake, m.get("time_uk","?")))
        else:
            if bet:
                print(f"  No edge passes filter '{bet_filter}' — skip")
            else:
                print(f"  No edge — skip")
            all_matches_info.append((home, away, m["line"], m["over_odds"], m["under_odds"], None, None, m.get("time_uk","?")))
        print()

    print(f"{'='*65}")
    if bets:
        print(f"  {tour.upper()} TODAY'S BETS ({len(bets)})")
        print(f"{'='*65}")
        total_stake = 0.0
        for home, away, best_of, bet, stake in bets:
            stake_str = f"£{stake:.2f}" if stake else "—"
            print(f"  • {home} vs {away}: {bet['side']} {bet['line']} @ {bet['odds']} "
                  f"| edge {bet['edge']}% | stake {stake_str}")
            if stake:
                total_stake += stake
        if bankroll and total_stake:
            print(f"\n  Total stake: £{total_stake:.2f} ({total_stake/bankroll*100:.1f}% of bankroll)")
    else:
        print(f"  NO {tour.upper()} BETS QUALIFY TODAY")
    print(f"{'='*65}")

    log_bets(bets, tour, surface, today)
    return all_matches_info


def main():
    parser = argparse.ArgumentParser(description="Barnett-Clarke Tennis Total Games Predictor")
    parser.add_argument("--atp",        action="store_true", help="Run ATP predictions")
    parser.add_argument("--wta",        action="store_true", help="Run WTA predictions")
    parser.add_argument("--filter",     default="both",
                        choices=["over", "under", "both", "under_opt"],
                        help="Bet filter (default: both)")
    parser.add_argument("--api-key",    default=os.environ.get("ODDS_API_KEY", ""))
    parser.add_argument("--bankroll",   type=float, default=None)
    parser.add_argument("--kelly",      type=float, default=0.25,
                        help="Kelly fraction (default: 0.25)")
    parser.add_argument("--max-stake",  type=float, default=0.05,
                        help="Max stake as fraction of bankroll (default: 0.05)")
    parser.add_argument("--surface",    default=None,
                        help="Override surface (Hard/Clay/Grass). Auto-detected by month if omitted.")
    parser.add_argument("--pnl",        action="store_true",
                        help="Show P&L summary only, no new predictions")
    args = parser.parse_args()

    # Default to ATP if neither flag given
    if not args.atp and not args.wta:
        args.atp = True

    tours = []
    if args.atp:
        tours.append("atp")
    if args.wta:
        tours.append("wta")

    # Surface auto-detect by month
    SURFACE_CALENDAR = {
        1: "Hard", 2: "Hard", 3: "Hard",
        4: "Clay", 5: "Clay", 6: "Clay",
        7: "Grass", 8: "Hard", 9: "Hard",
        10: "Hard", 11: "Hard", 12: "Hard",
    }
    surface = args.surface or SURFACE_CALENDAR.get(date.today().month, "Hard")

    if args.pnl:
        for tour in tours:
            print_pnl_summary(tour)
        return

    if not args.api_key:
        print("ERROR: No API key. Get one free at https://the-odds-api.com")
        print("Then run: python tennis/predict_today.py --atp --api-key YOUR_KEY")
        sys.exit(1)

    bankroll = args.bankroll
    if bankroll is None:
        try:
            bankroll = float(input("\nEnter your bankroll in £ (for Kelly sizing, or press Enter to skip): ").strip())
        except (ValueError, EOFError):
            bankroll = None

    # Build lookup per tour (WTA and ATP have different player pools)
    for tour in tours:
        print(f"\nLoading {tour.upper()} historical data ({START_YEAR}-{END_YEAR})...")
        df = load_data(START_YEAR, END_YEAR, tour=tour)
        print(f"Building {tour.upper()} serve profiles...")
        lookup = build_lookup(df)
        print(f"  {len(lookup)} player-surface profiles loaded")

        run_tour(tour, args.api_key, surface, args.filter,
                 bankroll, args.kelly, args.max_stake, lookup)
        print_pnl_summary(tour)

    # Generate HTML dashboard
    try:
        from live_dashboard import generate_dashboard
        tours_with_data = [t for t in tours if os.path.exists(_log_file(t))]
        if tours_with_data:
            path = generate_dashboard(tours_with_data)
            if path:
                print(f"\nDashboard saved: {path}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
