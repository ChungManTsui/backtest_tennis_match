"""
kelly_backtest.py — Run your own Kelly staking simulation on the tennis backtest results.

Usage:
    python tennis/kelly_backtest.py
    python tennis/kelly_backtest.py --kelly-fraction 0.25 --max-stake 0.05 --bankroll 1000
    python tennis/kelly_backtest.py --odds-min 1.8 --odds-max 2.0
    python tennis/kelly_backtest.py --surface Hard
    python tennis/kelly_backtest.py --bet-side Under
"""

import argparse
import pandas as pd
import numpy as np

CSV_PATH = "tennis/data/backtest_results.csv"


def simulate(df: pd.DataFrame, starting_bankroll: float,
             kelly_fraction: float, max_stake_pct: float) -> dict:
    bankroll = starting_bankroll
    peak = bankroll
    max_dd = 0.0
    history = []

    for _, row in df.iterrows():
        kelly = float(row["kelly"])
        odds  = float(row["bk_odds"])
        won   = bool(row["won"])

        kelly_capped = min(kelly, 1.0)  # cap raw Kelly at 100%
        stake = bankroll * min(kelly_capped * kelly_fraction, max_stake_pct)
        stake = max(stake, 0.0)

        pnl = stake * (odds - 1) if won else -stake
        bankroll += pnl
        bankroll = max(bankroll, 0.01)  # floor at 1p

        peak = max(peak, bankroll)
        dd = (peak - bankroll) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

        history.append({
            "date":      row["date"],
            "match":     f"{row['winner']} vs {row['loser']}",
            "surface":   row["surface"],
            "bet_side":  row["bet_side"],
            "odds":      round(odds, 3),
            "edge":      round(float(row["edge"]) * 100, 1),
            "model_prob": round(float(row["model_prob"]) * 100, 1),
            "kelly_full": round(float(row["kelly"]) * 100, 1),
            "stake":     round(stake, 2),
            "won":       won,
            "pnl":       round(pnl, 2),
            "bankroll":  round(bankroll, 2),
        })

    total = len(df)
    wins  = int(df["won"].sum())
    pnl   = bankroll - starting_bankroll

    return {
        "total_bets":     total,
        "wins":           wins,
        "win_rate":       round(wins / total * 100, 1) if total else 0,
        "starting":       starting_bankroll,
        "final_bankroll": round(bankroll, 2),
        "pnl":            round(pnl, 2),
        "roi":            round(pnl / starting_bankroll * 100, 1),
        "max_drawdown":   round(max_dd * 100, 1),
        "history":        history,
    }


def main():
    parser = argparse.ArgumentParser(description="Kelly staking backtest on tennis model results")
    parser.add_argument("--bankroll",      type=float, default=1000.0,  help="Starting bankroll £ (default: 1000)")
    parser.add_argument("--kelly-fraction",type=float, default=0.25,    help="Kelly fraction (default: 0.25 = Quarter Kelly)")
    parser.add_argument("--max-stake",     type=float, default=0.05,    help="Max stake as % of bankroll (default: 0.05 = 5%%)")
    parser.add_argument("--odds-min",      type=float, default=1.0,     help="Min odds filter (default: 1.0)")
    parser.add_argument("--odds-max",      type=float, default=99.0,    help="Max odds filter (default: no limit)")
    parser.add_argument("--surface",       type=str,   default=None,    help="Filter surface: Hard / Clay / Grass")
    parser.add_argument("--bet-side",      type=str,   default=None,    help="Filter bet side: Over / Under")
    parser.add_argument("--year-start",    type=int,   default=2015,    help="Start year (default: 2015)")
    parser.add_argument("--year-end",      type=int,   default=2024,    help="End year (default: 2024)")
    parser.add_argument("--reset-yearly",  action="store_true",         help="Reset bankroll to starting amount each year")
    parser.add_argument("--show-bets",     type=int,   default=0,       help="Show last N bets in detail (default: 0)")
    args = parser.parse_args()

    # Load data
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"ERROR: {CSV_PATH} not found. Run tennis/main.py first to generate backtest results.")
        return

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[(df["year"] >= args.year_start) & (df["year"] <= args.year_end)]

    # Apply filters
    df = df[(df["bk_odds"] >= args.odds_min) & (df["bk_odds"] <= args.odds_max)]
    if args.surface:
        df = df[df["surface"].str.lower() == args.surface.lower()]
    if args.bet_side:
        df = df[df["bet_side"].str.lower() == args.bet_side.lower()]

    df = df.sort_values("date").reset_index(drop=True)

    if df.empty:
        print("No bets match your filters.")
        return

    print("=" * 65)
    print("  TENNIS MODEL — KELLY STAKING BACKTEST")
    print("=" * 65)
    print(f"  Kelly fraction : {args.kelly_fraction} ({args.kelly_fraction*100:.0f}% of full Kelly)")
    print(f"  Max stake      : {args.max_stake*100:.0f}% of bankroll")
    print(f"  Starting £     : £{args.bankroll:,.2f}")
    print(f"  Years          : {args.year_start}–{args.year_end}")
    print(f"  Odds filter    : {args.odds_min}–{args.odds_max}")
    if args.surface:
        print(f"  Surface        : {args.surface}")
    if args.bet_side:
        print(f"  Bet side       : {args.bet_side}")
    print()

    if args.reset_yearly:
        # Year-by-year, reset bankroll each year
        print(f"  {'Year':<6} {'Bets':>5} {'Won':>5} {'Win%':>6} {'P&L':>10} {'ROI':>8} {'Final £':>10} {'MaxDD':>7}")
        print(f"  {'-'*6} {'-'*5} {'-'*5} {'-'*6} {'-'*10} {'-'*8} {'-'*10} {'-'*7}")
        total_pnl = 0
        for year in range(args.year_start, args.year_end + 1):
            ydf = df[df["year"] == year].copy()
            if ydf.empty:
                continue
            s = simulate(ydf, args.bankroll, args.kelly_fraction, args.max_stake)
            total_pnl += s["pnl"]
            print(f"  {year:<6} {s['total_bets']:>5} {s['wins']:>5} {s['win_rate']:>5.1f}% "
                  f"£{s['pnl']:>+9.2f} {s['roi']:>+7.1f}% "
                  f"£{s['final_bankroll']:>9.2f} {s['max_drawdown']:>6.1f}%")
        print(f"\n  Total P&L across all years: £{total_pnl:+,.2f}")
        print(f"  Avg P&L per year: £{total_pnl/(args.year_end - args.year_start + 1):+,.2f}")
    else:
        # Single continuous run
        s = simulate(df, args.bankroll, args.kelly_fraction, args.max_stake)
        print(f"  Total bets     : {s['total_bets']}")
        print(f"  Won            : {s['wins']} ({s['win_rate']}%)")
        print(f"  Starting       : £{s['starting']:,.2f}")
        print(f"  Final bankroll : £{s['final_bankroll']:,.2f}")
        print(f"  P&L            : £{s['pnl']:+,.2f}")
        print(f"  ROI            : {s['roi']:+.1f}%")
        print(f"  Max drawdown   : {s['max_drawdown']:.1f}%")

        # Year breakdown
        print(f"\n  Year-by-year breakdown (continuous bankroll):")
        print(f"  {'Year':<6} {'Bets':>5} {'Won':>5} {'Win%':>6} {'End Bankroll':>14}")
        print(f"  {'-'*6} {'-'*5} {'-'*5} {'-'*6} {'-'*14}")

        history_df = pd.DataFrame(s["history"])
        history_df["year"] = pd.to_datetime(history_df["date"]).dt.year
        for year, g in history_df.groupby("year"):
            bets = len(g)
            wins = int(g["won"].sum())
            end_bank = g["bankroll"].iloc[-1]
            print(f"  {year:<6} {bets:>5} {wins:>5} {wins/bets*100:>5.1f}% £{end_bank:>13,.2f}")

        if args.show_bets > 0:
            print(f"\n  Last {args.show_bets} bets:")
            print(f"  {'Date':<12} {'Match':<40} {'Side':<6} {'Odds':>5} {'Edge%':>6} {'Stake':>8} {'W/L':>4} {'P&L':>8} {'Bank':>10}")
            print(f"  {'-'*12} {'-'*40} {'-'*6} {'-'*5} {'-'*6} {'-'*8} {'-'*4} {'-'*8} {'-'*10}")
            for h in s["history"][-args.show_bets:]:
                result = "W" if h["won"] else "L"
                match_str = h["match"][:39]
                print(f"  {h['date']:<12} {match_str:<40} {h['bet_side']:<6} {h['odds']:>5.3f} "
                      f"{h['edge']:>5.1f}% £{h['stake']:>7.2f} {result:>4} £{h['pnl']:>+7.2f} £{h['bankroll']:>9.2f}")

    print()
    print("=" * 65)
    print("  TIP: Try different settings:")
    print("  python tennis/kelly_backtest.py --kelly-fraction 0.5 --max-stake 0.08")
    print("  python tennis/kelly_backtest.py --bet-side Under --reset-yearly")
    print("  python tennis/kelly_backtest.py --surface Clay --show-bets 20")
    print("=" * 65)


if __name__ == "__main__":
    main()
