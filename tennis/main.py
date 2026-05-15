"""
main.py — Barnett-Clarke Tennis Total Games Backtest

Run:
    python tennis/main.py
    python tennis/main.py --bankroll 500
    python tennis/main.py --bankroll 100 --kelly-fraction 0.25 --max-stake 0.05
    python tennis/main.py --bankroll 100 --tour wta

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
from betfair_matcher import load_betfair_odds, match_odds_to_sackmann
from report import generate_html

START_YEAR = 2010
END_YEAR   = 2026


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
    parser.add_argument("--tour",           type=str,   default="atp",
                        choices=["atp", "wta"],
                        help="Tour to backtest: atp or wta (default: atp)")
    parser.add_argument("--reset-yearly",   action="store_true",
                        help="Reset bankroll back to starting amount each year (no compounding across years)")
    parser.add_argument("--real-odds",      action="store_true",
                        help="Use real Betfair TOTAL_GAMES closing odds (requires betfair_total_games_*.csv in tennis/data/)")
    args = parser.parse_args()

    tour_label = args.tour.upper()
    print("=" * 70)
    print(f"  BARNETT-CLARKE TENNIS TOTAL GAMES MODEL — {tour_label}")
    print(f"  Data: {tour_label} {START_YEAR}-{END_YEAR} | Grand Slams + Masters + 500/250")
    print(f"  Bankroll: £{args.bankroll:,.2f} | Kelly: {args.kelly_fraction*100:.0f}% | Max stake: {args.max_stake*100:.0f}% | Vig: {args.vig*100:.0f}%")
    print("=" * 70)

    print(f"\nDownloading {tour_label} match data...")
    df = load_data(START_YEAR, END_YEAR, tour=args.tour)

    print("\nBuilding rolling serve stats (walk-forward, no lookahead)...")
    df = build_serve_stats(df)

    n_with_serve = df[["p_hold_winner", "p_hold_loser"]].dropna().shape[0]
    print(f"  Matches with serve data: {n_with_serve} / {len(df)}")

    # Optionally merge real Betfair odds
    use_real_odds = False
    suffix = ""
    if args.real_odds:
        bf_csv = os.path.join(os.path.dirname(__file__), "data", f"betfair_total_games_{args.tour}.csv")
        if os.path.exists(bf_csv):
            print(f"\nLoading real Betfair odds from {bf_csv}...")
            bf_df = load_betfair_odds(bf_csv)
            print(f"  {len(bf_df)} Betfair TOTAL_GAMES records loaded")
            print("  Matching to Sackmann data...")
            df = match_odds_to_sackmann(df, bf_df)
            use_real_odds = True
            suffix = "_real"
        else:
            print(f"\n  WARNING: --real-odds requested but {bf_csv} not found.")
            print(f"  Run: python tennis/betfair_odds_extractor.py first.")

    print("\nRunning backtest...")
    results_both     = backtest(df, starting_bankroll=args.bankroll, vig=args.vig, bet_filter="both",      use_real_odds=use_real_odds)
    results_over     = backtest(df, starting_bankroll=args.bankroll, vig=args.vig, bet_filter="over",      use_real_odds=use_real_odds)
    results_under    = backtest(df, starting_bankroll=args.bankroll, vig=args.vig, bet_filter="under",     use_real_odds=use_real_odds)
    results_under_opt= backtest(df, starting_bankroll=args.bankroll, vig=args.vig, bet_filter="under_opt", use_real_odds=use_real_odds)

    for label, results, bet_filter in [
        ("OVER + UNDER (both)",              results_both,      "both"),
        ("OVER ONLY",                        results_over,      "over"),
        ("UNDER ONLY",                       results_under,     "under"),
        ("UNDER OPTIMISED (bo5 + gap>0)",    results_under_opt, "under_opt"),
    ]:
        bets_df = results["bets_df"]
        print(f"\n{'='*70}")
        print(f"  {label} ({len(bets_df)} bets)")
        print(f"{'='*70}")
        if bets_df.empty:
            print("  No qualifying bets found.")
            continue

        print_summary("OVERALL", results["summary"])

        print(f"\n  By Tournament Level:")
        for lvl in ["Grand Slam", "Masters 1000", "ATP 500/250", "ATP Finals"]:
            if lvl in results["by_level"]:
                print_summary(lvl, results["by_level"][lvl])

        print(f"\n  By Surface:")
        for surf in ["Hard", "Clay", "Grass"]:
            if surf in results["by_surface"]:
                print_summary(surf, results["by_surface"][surf])

        simulate_bankroll_yearly(bets_df, args.bankroll, args.kelly_fraction, args.max_stake,
                                 reset_yearly=args.reset_yearly)

        cal = results.get("calibration", [])
        if cal:
            print(f"\n  Model Calibration (predicted vs actual win rate):")
            print(f"  {'Bucket':<12} {'Bets':>5} {'Predicted%':>10} {'Actual%':>8} {'Diff':>6}")
            print(f"  {'-'*12} {'-'*5} {'-'*10} {'-'*8} {'-'*6}")
            for r in cal:
                flag = " ok" if abs(r["diff"]) <= 3 else " !!"
                print(f"  {r['prob_bucket']:<12} {r['bets']:>5} {r['predicted_%']:>9.1f}% {r['actual_%']:>7.1f}% {r['diff']:>+5.1f}%{flag}")
            print(f"  (ok = well calibrated within 3%, !! = miscalibrated)")

        out = f"tennis/data/backtest_results_{args.tour}_{bet_filter}{suffix}.csv"
        bets_df.to_csv(out, index=False)
        print(f"\n  Bet log saved to: {out}")

    print("=" * 70)

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    report_path = generate_html(args.tour, args.bankroll, args.kelly_fraction, args.max_stake, data_dir, suffix=suffix)
    print(f"\nHTML report saved to: {report_path}")


if __name__ == "__main__":
    main()
