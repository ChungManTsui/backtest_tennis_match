"""
scheduler.py — Daily runner with built-in 9am UK loop.

Run once:       python3 tennis/scheduler.py
Run forever:    screen -dmS tennis python3 tennis/scheduler.py --loop
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta
from collections import defaultdict

from data_loader import load_data
from serve_model import build_serve_stats, compute_serve_hold_pct
from markov import games_distribution_fast, prob_hold_game
from telegram_bot import send_predictions, send_pnl_update, send_heartbeat

# ── Config ────────────────────────────────────────────────────────────────────
ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "")
START_YEAR    = 2015
END_YEAR      = 2025
MIN_EDGE      = 0.04
LOG_FILE       = "tennis/data/bet_log.csv"
BANKROLL_FILE  = "tennis/data/bankroll.json"
LOG_COLS      = ["date", "surface", "home", "away", "bet_side", "line",
                 "odds", "edge", "model_prob", "stake", "result", "pnl"]
ODDS_API_URL  = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
SURFACE_CALENDAR = {
    1: "Hard", 2: "Hard", 3: "Hard",    # Jan–Mar: Australian Open season
    4: "Clay", 5: "Clay", 6: "Clay",    # Apr–Jun: Clay season / Roland Garros
    7: "Grass", 8: "Hard", 9: "Hard",   # Jul: Wimbledon, Aug–Sep: US Open
    10: "Hard", 11: "Hard", 12: "Hard", # Oct–Dec: Indoor hard
}
# ─────────────────────────────────────────────────────────────────────────────


def get_surface() -> str:
    return SURFACE_CALENDAR.get(date.today().month, "Hard")


def fetch_active_atp_sports() -> list[str]:
    try:
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports",
            params={"apiKey": ODDS_API_KEY},
            timeout=10,
        )
        r.raise_for_status()
        return [s["key"] for s in r.json() if s.get("active") and "tennis_atp" in s["key"]]
    except Exception as e:
        print(f"  Could not fetch active sports: {e}")
        return []


def fetch_odds() -> list[dict]:
    sports = fetch_active_atp_sports()
    if not sports:
        return []
    all_events = []
    for sport in sports:
        try:
            r = requests.get(ODDS_API_URL.format(sport=sport), params={
                "apiKey": ODDS_API_KEY, "regions": "uk,eu",
                "markets": "totals", "oddsFormat": "decimal",
            }, timeout=10)
            if r.status_code in (404, 422):
                continue
            r.raise_for_status()
            data = r.json()
            if data:
                all_events.extend(data)
                print(f"  {sport}: {len(data)} match(es)")
        except Exception:
            continue
    return all_events


def parse_events(events: list[dict]) -> list[dict]:
    from datetime import timezone
    import zoneinfo
    uk_tz = zoneinfo.ZoneInfo("Europe/London")
    today_uk = date.today()  # system date, but we'll compare in UK time
    parsed = []
    for ev in events:
        # commence_time is UTC e.g. "2026-05-10T08:00:00Z" — convert to UK time
        commence = ev.get("commence_time", "")
        if not commence:
            continue
        try:
            utc_dt = pd.Timestamp(commence, tz="UTC")
            uk_dt  = utc_dt.tz_convert(uk_tz)
            if uk_dt.date() != today_uk:
                continue
        except Exception:
            continue

        home = ev.get("home_team", "")
        away = ev.get("away_team", "")
        bookmakers = ev.get("bookmakers", [])
        bk = next((b for b in bookmakers if b["key"] == "bet365"), None)
        if bk is None:
            bk = next((b for b in bookmakers if b["key"] == "pinnacle"), None)
        if bk is None:
            bk = next((b for b in bookmakers if b.get("markets")), None)
        if bk is None:
            continue
        for market in bk.get("markets", []):
            if market["key"] != "totals":
                continue
            outcomes = {o["name"]: o for o in market.get("outcomes", [])}
            if "Over" not in outcomes or "Under" not in outcomes:
                continue
            parsed.append({
                "home":       home,
                "away":       away,
                "line":       float(outcomes["Over"]["point"]),
                "over_odds":  float(outcomes["Over"]["price"]),
                "under_odds": float(outcomes["Under"]["price"]),
                "bookmaker":  bk["key"],
                "time_uk":    uk_dt.strftime("%H:%M"),
            })
    return parsed


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


def predict(home, away, line, over_odds, under_odds, surface, lookup):
    p_w = fuzzy_lookup(home, surface, lookup)
    p_l = fuzzy_lookup(away, surface, lookup)
    if p_w is None or p_l is None:
        return None
    p_w = min(max(p_w, 0.35), 0.95)
    p_l = min(max(p_l, 0.35), 0.95)
    try:
        dist = games_distribution_fast(p_w, p_l, 3)
    except Exception:
        return None
    if not dist:
        return None
    our_p_over  = sum(p for g, p in dist.items() if g > line)
    our_p_under = sum(p for g, p in dist.items() if g < line)
    imp_over    = 1.0 / over_odds
    imp_under   = 1.0 / under_odds
    edge_over   = our_p_over  - imp_over
    edge_under  = our_p_under - imp_under
    mean_g      = sum(g * p for g, p in dist.items())

    if edge_over > MIN_EDGE and edge_over >= edge_under:
        b = over_odds - 1
        kelly = (b * our_p_over - (1 - our_p_over)) / b if b > 0 else 0
        return {"side": "Over", "line": line, "odds": over_odds,
                "model_prob": round(our_p_over, 4), "edge": round(edge_over * 100, 1),
                "kelly": round(kelly, 4), "mean_games": round(mean_g, 1)}
    elif edge_under > MIN_EDGE:
        b = under_odds - 1
        kelly = (b * our_p_under - (1 - our_p_under)) / b if b > 0 else 0
        return {"side": "Under", "line": line, "odds": under_odds,
                "model_prob": round(our_p_under, 4), "edge": round(edge_under * 100, 1),
                "kelly": round(kelly, 4), "mean_games": round(mean_g, 1)}
    return None


def log_bets(bets: list, surface: str, today: str):
    os.makedirs("tennis/data", exist_ok=True)
    if not bets:
        return
    rows = []
    for home, away, bet, stake in bets:
        rows.append({
            "date": today, "surface": surface,
            "home": home, "away": away,
            "bet_side": bet["side"], "line": bet["line"],
            "odds": bet["odds"], "edge": bet["edge"],
            "model_prob": bet["model_prob"],
            "stake": round(stake, 2) if stake else "",
            "result": "", "pnl": "",
        })
    new_df = pd.DataFrame(rows, columns=LOG_COLS)
    if os.path.exists(LOG_FILE):
        existing = pd.read_csv(LOG_FILE)
        existing_keys = set(zip(existing["date"], existing["home"], existing["away"]))
        new_df = new_df[~new_df.apply(
            lambda r: (r["date"], r["home"], r["away"]) in existing_keys, axis=1)]
        if not new_df.empty:
            pd.concat([existing, new_df], ignore_index=True).to_csv(LOG_FILE, index=False)
    else:
        new_df.to_csv(LOG_FILE, index=False)
    print(f"  {len(new_df)} bet(s) logged to {LOG_FILE}")


def load_bankroll(default: float = 100.0) -> float:
    import json
    if os.path.exists(BANKROLL_FILE):
        with open(BANKROLL_FILE) as f:
            return float(json.load(f)["bankroll"])
    return default


def save_bankroll(bankroll: float):
    import json
    os.makedirs(os.path.dirname(BANKROLL_FILE), exist_ok=True)
    with open(BANKROLL_FILE, "w") as f:
        json.dump({"bankroll": round(bankroll, 2)}, f)


def fetch_results_and_update():
    """
    Auto-fetch yesterday's ATP results from Sackmann live feed or API
    and fill in W/L in bet_log.csv automatically.
    Uses The Odds API scores endpoint.
    """
    if not os.path.exists(LOG_FILE):
        return
    df = pd.read_csv(LOG_FILE)
    pending = df[df["result"].astype(str).str.strip() == ""]
    if pending.empty:
        return

    yesterday = str(date.today() - timedelta(days=1))
    pending_yesterday = pending[pending["date"] == yesterday]
    if pending_yesterday.empty:
        return

    print(f"  Fetching results for {yesterday}...")
    sports = fetch_active_atp_sports()

    scores_map = {}
    for sport in sports:
        try:
            r = requests.get(
                f"https://api.the-odds-api.com/v4/sports/{sport}/scores",
                params={"apiKey": ODDS_API_KEY, "daysFrom": 1},
                timeout=10,
            )
            if r.status_code != 200:
                continue
            for ev in r.json():
                if not ev.get("completed"):
                    continue
                home = ev.get("home_team", "").lower()
                away = ev.get("away_team", "").lower()
                scores = ev.get("scores", [])
                if scores:
                    score_map = {s["name"].lower(): s["score"] for s in scores}
                    scores_map[(home, away)] = score_map
        except Exception:
            continue

    updated = 0
    for idx, row in pending_yesterday.iterrows():
        home_key = row["home"].lower()
        away_key = row["away"].lower()
        score_data = scores_map.get((home_key, away_key)) or scores_map.get((away_key, home_key))
        if not score_data:
            continue

        try:
            total_games = sum(int(v) for v in score_data.values())
        except Exception:
            continue

        won = (total_games > row["line"]) if row["bet_side"] == "Over" else (total_games < row["line"])
        result = "W" if won else "L"
        stake = float(row["stake"]) if str(row["stake"]).strip() != "" else 0
        pnl = round(stake * (float(row["odds"]) - 1), 2) if won else round(-stake, 2)

        df.at[idx, "result"] = result
        df.at[idx, "pnl"] = pnl
        updated += 1
        print(f"  Auto-result: {row['home']} vs {row['away']} → {total_games} games → {result} (£{pnl:+.2f})")

    if updated:
        df.to_csv(LOG_FILE, index=False)
        print(f"  {updated} result(s) auto-filled.")
        # compound bankroll: add yesterday's P&L
        yesterday_pnl = df[df["date"] == str(date.today() - timedelta(days=1))]["pnl"]
        yesterday_pnl = pd.to_numeric(yesterday_pnl, errors="coerce").fillna(0).sum()
        new_bankroll = round(load_bankroll() + yesterday_pnl, 2)
        save_bankroll(new_bankroll)
        print(f"  Bankroll updated: £{load_bankroll() - yesterday_pnl:.2f} → £{new_bankroll:.2f} ({yesterday_pnl:+.2f})")



def get_pnl_summary() -> dict:
    if not os.path.exists(LOG_FILE):
        return {}
    df = pd.read_csv(LOG_FILE)
    settled = df[df["result"].astype(str).str.strip().str.upper().isin(["W", "L"])].copy()
    if settled.empty:
        return {}
    settled["pnl"]   = pd.to_numeric(settled["pnl"],   errors="coerce").fillna(0)
    settled["stake"] = pd.to_numeric(settled["stake"], errors="coerce").fillna(0)
    total_bets   = len(settled)
    wins         = int((settled["result"].str.upper() == "W").sum())
    total_staked = settled["stake"].sum()
    total_pnl    = settled["pnl"].sum()
    return {
        "total_bets":   total_bets,
        "wins":         wins,
        "win_rate":     round(wins / total_bets * 100, 1) if total_bets else 0,
        "total_staked": round(total_staked, 2),
        "total_pnl":    round(total_pnl, 2),
        "roi":          round(total_pnl / total_staked * 100, 1) if total_staked else 0,
    }


def _ask(prompt: str, default: str) -> str:
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else default


def main():
    today   = str(date.today())
    surface = get_surface()

    # Ask for settings if not set as env vars
    bankroll       = load_bankroll(float(os.environ.get("BANKROLL", "100")))
    kelly_fraction = float(os.environ.get("KELLY_FRACTION") or _ask("Kelly fraction (e.g. 0.25 = Quarter Kelly)", "0.25"))
    max_stake      = float(os.environ.get("MAX_STAKE")      or _ask("Max stake per bet (e.g. 0.05 = 5%)", "0.05"))

    print("=" * 60)
    print(f"  ATP TENNIS SCHEDULER — {today}")
    print(f"  Surface: {surface} | Bankroll: £{bankroll} | Kelly: {kelly_fraction*100:.0f}% | Max stake: {max_stake*100:.0f}%")
    print("=" * 60)

    # Step 1 — Auto-fill yesterday's results
    print("\nChecking yesterday's results...")
    try:
        fetch_results_and_update()
    except Exception as e:
        print(f"  Result fetch failed: {e}")

    # Step 2 — Send P&L update
    pnl = get_pnl_summary()
    if pnl:
        print(f"\n  Running P&L: {pnl['wins']}/{pnl['total_bets']} won | "
              f"ROI {pnl['roi']:+.1f}% | P&L £{pnl['total_pnl']:+.2f}")
        send_pnl_update(pnl, today)

    # Step 3 — Load data and build profiles
    print("\nLoading ATP data...")
    df = load_data(START_YEAR, END_YEAR)
    print("Building serve profiles...")
    lookup = build_lookup(df)
    print(f"  {len(lookup)} profiles loaded")

    # Step 4 — Fetch today's odds
    print("\nFetching today's odds...")
    if not ODDS_API_KEY:
        print("  ERROR: ODDS_API_KEY not set.")
        send_heartbeat(today, "ERROR: No API key")
        return

    events  = fetch_odds()
    matches = parse_events(events)
    print(f"  {len(matches)} match(es) with totals lines")

    # Step 5 — Predict
    bets = []
    all_matches = []
    for m in matches:
        bet = predict(m["home"], m["away"], m["line"],
                      m["over_odds"], m["under_odds"], surface, lookup)
        stake = None
        if bet:
            stake = bankroll * min(bet["kelly"] * kelly_fraction, max_stake)
            bets.append((m["home"], m["away"], bet, stake))
            print(f"  ★ {m['home']} vs {m['away']}: {bet['side']} {bet['line']} "
                  f"@ {bet['odds']} | edge {bet['edge']}% | £{stake:.2f}")
        else:
            print(f"  — {m['home']} vs {m['away']}: no edge")
        all_matches.append((m["home"], m["away"], m["line"],
                            m["over_odds"], m["under_odds"], bet, stake, m.get("time_uk", "?")))

    # Step 6 — Log bets
    log_bets(bets, surface, today)

    # Step 7 — Telegram alert
    send_predictions(all_matches, bets, surface, today, bankroll,
                     kelly_fraction=kelly_fraction, max_stake=max_stake)

    print(f"\n  Done. {len(bets)} bet(s) sent to Telegram.")
    print("=" * 60)


def loop():
    """Run main() every day at 9am UK time. Use with: screen -dmS tennis python3 tennis/scheduler.py --loop"""
    import time
    import zoneinfo
    uk_tz = zoneinfo.ZoneInfo("Europe/London")
    print("[scheduler] Loop mode started. Will run daily at 09:00 UK time.")
    while True:
        import datetime as dt
        now_uk     = dt.datetime.now(uk_tz)
        target     = now_uk.replace(hour=9, minute=0, second=0, microsecond=0)
        if now_uk >= target:
            target += dt.timedelta(days=1)
        wait = (target - now_uk).total_seconds()
        print(f"[scheduler] Sleeping {wait/3600:.1f}h until {target.strftime('%Y-%m-%d 09:00 UK')}")
        time.sleep(wait)
        try:
            main()
        except Exception as e:
            print(f"[scheduler] ERROR: {e}")
        time.sleep(60)  # avoid double-firing


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="Run forever, firing at 9am UK time daily")
    args = parser.parse_args()
    if args.loop:
        loop()
    else:
        main()
