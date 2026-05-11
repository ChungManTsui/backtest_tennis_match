"""
main.py — Barnett-Clarke Tennis Total Games Backtest

Run:
    python tennis/main.py
    python tennis/main.py --bankroll 500
    python tennis/main.py --bankroll 100 --kelly-fraction 0.25 --max-stake 0.05
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from data_loader import load_data
from serve_model import build_serve_stats
from backtest import backtest

START_YEAR = 2015
END_YEAR   = 2025


def print_summary(label: str, s: dict):
    if not s or s["total_bets"] == 0:
        print(f"  {label}: No bets")
        return
    print(f"  {label:<18} | Bets: {s['total_bets']:>4} | "
          f"Win%: {s['win_rate']:>5.1f}% (BE: {s['breakeven']:.1f}%) | "
          f"ROI: {s['roi']:>+6.1f}% | "
          f"P&L: £{s['pnl']:>+8.2f} | "
          f"MaxDD: {s['max_drawdown']:.1f}%")


def simulate_bankroll_yearly(bets_df: pd.DataFrame, starting_bankroll: float,
                              kelly_fraction: float, max_stake_pct: float,
                              reset_yearly: bool = False):
    """Compound Kelly simulation year-by-year. Optionally reset bankroll each year."""
    mode = "reset each year" if reset_yearly else "compounding across years"
    print(f"\n  Year-by-Year Breakdown (Kelly {kelly_fraction*100:.0f}%, "
          f"{max_stake_pct*100:.0f}% max stake, £{starting_bankroll:,.2f} start, {mode}):")
    print(f"  {'Year':<6} {'Bets':>5} {'Won':>5} {'Win%':>6} {'ROI':>8} {'P&L':>10} {'Bankroll':>10} {'MaxDD':>7}")
    print(f"  {'-'*6} {'-'*5} {'-'*5} {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*7}")

    bets_df = bets_df.copy()
    bets_df["year"] = pd.to_datetime(bets_df["date"]).dt.year

    bankroll = starting_bankroll
    peak = bankroll
    total_pnl_reset = 0.0

    for year in sorted(bets_df["year"].unique()):
        if reset_yearly:
            bankroll = starting_bankroll
        yb = bets_df[bets_df["year"] == year]
        wins = int(yb["won"].sum())
        n = len(yb)
        start_bankroll = bankroll
        year_peak = bankroll
        year_max_dd = 0.0

        for _, bet in yb.iterrows():
            stake = bankroll * min(bet["kelly"] * kelly_fraction, max_stake_pct)
            stake = max(stake, 0.0)
            if bet["won"]:
                bankroll += stake * (bet["bk_odds"] - 1)
            else:
                bankroll -= stake
            bankroll = max(bankroll, 0.01)
            year_peak = max(year_peak, bankroll)
            peak = max(peak, bankroll)
            dd = (year_peak - bankroll) / year_peak if year_peak > 0 else 0
            year_max_dd = max(year_max_dd, dd)

        pnl = bankroll - start_bankroll
        roi = pnl / start_bankroll * 100
        total_pnl_reset += pnl
        print(f"  {year:<6} {n:>5} {wins:>5} {wins/n*100:>5.1f}% "
              f"{roi:>+7.1f}% £{pnl:>+9.2f} £{bankroll:>9.2f} {year_max_dd*100:>6.1f}%")

    if reset_yearly:
        avg_roi = total_pnl_reset / starting_bankroll * 100 / len(bets_df["year"].unique())
        print(f"\n  Starting bankroll : £{starting_bankroll:,.2f} (reset each year)")
        print(f"  Total P&L         : £{total_pnl_reset:+,.2f} (sum of all years)")
        print(f"  Avg ROI per year  : {avg_roi:+.1f}%")
    else:
        total_pnl = bankroll - starting_bankroll
        total_roi = total_pnl / starting_bankroll * 100
        overall_dd = (peak - bankroll) / peak * 100 if peak > 0 else 0
        print(f"\n  Starting bankroll : £{starting_bankroll:,.2f}")
        print(f"  Final bankroll    : £{bankroll:,.2f}")
        print(f"  Total P&L         : £{total_pnl:+,.2f}")
        print(f"  Total ROI         : {total_roi:+.1f}%")
        print(f"  Max drawdown      : {overall_dd:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Barnett-Clarke Tennis Backtest")
    parser.add_argument("--bankroll",       type=float, default=1000.0,
                        help="Starting bankroll in £ (default: 1000)")
    parser.add_argument("--kelly-fraction", type=float, default=0.25,
                        help="Kelly fraction (default: 0.25 = Quarter Kelly)")
    parser.add_argument("--max-stake",      type=float, default=0.05,
                        help="Max stake as fraction of bankroll (default: 0.05 = 5%%)")
    parser.add_argument("--vig",            type=float, default=0.09,
                        help="Bookmaker margin on each side (default: 0.09 = 9%%)")
    parser.add_argument("--reset-yearly",   action="store_true",
                        help="Reset bankroll back to starting amount each year (no compounding across years)")
    args = parser.parse_args()

    print("=" * 70)
    print("  BARNETT-CLARKE TENNIS TOTAL GAMES MODEL")
    print(f"  Data: ATP {START_YEAR}–{END_YEAR} | Grand Slams + Masters + ATP 500/250")
    print(f"  Bankroll: £{args.bankroll:,.2f} | Kelly: {args.kelly_fraction*100:.0f}% | Max stake: {args.max_stake*100:.0f}% | Vig: {args.vig*100:.0f}%")
    print("=" * 70)

    print("\nDownloading ATP match data...")
    df = load_data(START_YEAR, END_YEAR)

    print("\nBuilding rolling serve stats (walk-forward, no lookahead)...")
    df = build_serve_stats(df)

    n_with_serve = df[["p_hold_winner", "p_hold_loser"]].dropna().shape[0]
    print(f"  Matches with serve data: {n_with_serve} / {len(df)}")

    print("\nRunning backtest...")
    results = backtest(df, starting_bankroll=args.bankroll, vig=args.vig)

    bets_df = results["bets_df"]
    if bets_df.empty:
        print("  No qualifying bets found. Try lowering MIN_EDGE in backtest.py.")
        return

    print(f"\n{'='*70}")
    print(f"  OVERALL RESULTS ({len(bets_df)} bets)")
    print(f"{'='*70}")
    print_summary("OVERALL", results["summary"])

    print(f"\n  By Tournament Level:")
    level_order = ["Grand Slam", "Masters 1000", "ATP 500/250", "ATP Finals"]
    for lvl in level_order:
        if lvl in results["by_level"]:
            print_summary(lvl, results["by_level"][lvl])

    print(f"\n  By Surface:")
    for surf in ["Hard", "Clay", "Grass"]:
        if surf in results["by_surface"]:
            print_summary(surf, results["by_surface"][surf])

    # Year-by-year compounding Kelly simulation
    simulate_bankroll_yearly(bets_df, args.bankroll, args.kelly_fraction, args.max_stake,
                             reset_yearly=args.reset_yearly)

    # Save results
    out = "tennis/data/backtest_results.csv"
    bets_df.to_csv(out, index=False)
    print(f"\n  Full bet log saved to: {out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
