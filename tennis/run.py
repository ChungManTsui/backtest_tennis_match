"""
run.py — Interactive launcher for the ATP Tennis Scheduler.

Usage:
    python3 tennis/run.py          # asks questions, runs once
    python3 tennis/run.py --loop   # asks questions, then runs every day at 9am UK time
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
except ImportError:
    pass


def ask(prompt: str, default: str) -> str:
    val = input(f"  {prompt} [{default}]: ").strip()
    return val if val else default


def get_settings() -> dict:
    print("=" * 60)
    print("  ATP TENNIS SCHEDULER — Setup")
    print("=" * 60)

    # API keys — read from env only, never ask
    odds_key   = os.environ.get("ODDS_API_KEY", "")
    tg_token   = os.environ.get("TELEGRAM_TOKEN", "")
    tg_chat    = os.environ.get("TELEGRAM_CHAT_ID", "")

    # Bankroll — only ask on first run, then saved to bankroll.json
    import json
    bankroll_file = os.path.join(os.path.dirname(__file__), "data", "bankroll.json")
    if os.path.exists(bankroll_file):
        with open(bankroll_file) as f:
            bankroll = str(json.load(f)["bankroll"])
        print(f"  Bankroll loaded from file: £{bankroll}")
    else:
        bankroll = ask("Your bankroll (£) — saved for future runs", "100")
        os.makedirs(os.path.dirname(bankroll_file), exist_ok=True)
        with open(bankroll_file, "w") as f:
            json.dump({"bankroll": round(float(bankroll), 2)}, f)
        print(f"  Bankroll £{bankroll} saved to {bankroll_file}")

    kelly_fraction = ask("Kelly fraction  (0.1=safe / 0.25=standard / 0.5=aggressive)", "0.25")
    max_stake      = ask("Max stake per bet (0.05 = 5%)", "0.05")

    settings = {
        "ODDS_API_KEY":     odds_key,
        "TELEGRAM_TOKEN":   tg_token,
        "TELEGRAM_CHAT_ID": tg_chat,
        "BANKROLL":         bankroll,
        "KELLY_FRACTION":   kelly_fraction,
        "MAX_STAKE":        max_stake,
    }

    print()
    print("  Settings confirmed:")
    print(f"    Bankroll   : £{bankroll}")
    print(f"    Kelly      : {float(kelly_fraction)*100:.0f}%")
    print(f"    Max stake  : {float(max_stake)*100:.0f}%")
    print(f"    Odds API   : {'✓ set' if odds_key else '✗ MISSING'}")
    print(f"    Telegram   : {'✓ set' if tg_token else '✗ MISSING'}")
    print("=" * 60)

    if not odds_key:
        print("  ERROR: ODDS_API_KEY not set in environment.")
        sys.exit(1)
    if not tg_token or not tg_chat:
        print("  WARNING: Telegram not configured — predictions won't be sent.")

    return settings


def apply_settings(settings: dict):
    for key, val in settings.items():
        os.environ[key] = str(val)


DAILY_HOUR = 8  # 8:00 AM UK


def run_loop(settings: dict):
    import zoneinfo
    import datetime as dt

    apply_settings(settings)
    uk_tz = zoneinfo.ZoneInfo("Europe/London")

    print(f"[run.py] Loop started — will fire every day at {DAILY_HOUR:02d}:00 UK time.")
    print("[run.py] Press Ctrl+C to stop.\n")

    from scheduler import main as run_scheduler

    # Fire immediately on first run
    print(f"[{dt.datetime.now(uk_tz).strftime('%Y-%m-%d %H:%M')} UK] Running today's predictions...")
    try:
        run_scheduler()
    except Exception as e:
        print(f"[run.py] ERROR: {e}")

    while True:
        now_uk = dt.datetime.now(uk_tz)
        target = now_uk.replace(hour=DAILY_HOUR, minute=0, second=0, microsecond=0)
        if now_uk >= target:
            target += dt.timedelta(days=1)

        wait = (target - now_uk).total_seconds()
        print(f"[{now_uk.strftime('%Y-%m-%d %H:%M')} UK] Sleeping {wait/3600:.1f}h until {DAILY_HOUR:02d}:00 UK...")

        try:
            time.sleep(wait)
        except KeyboardInterrupt:
            print("\n[run.py] Stopped.")
            sys.exit(0)

        print(f"[{dt.datetime.now(uk_tz).strftime('%Y-%m-%d %H:%M')} UK] Running scheduler...")
        try:
            run_scheduler()
        except Exception as e:
            print(f"[run.py] ERROR: {e}")

        time.sleep(60)  # avoid double-firing


def run_once(settings: dict):
    apply_settings(settings)
    from scheduler import main as run_scheduler
    run_scheduler()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ATP Tennis Scheduler")
    parser.add_argument("--loop", action="store_true",
                        help="Run every day at 9am UK time (use with screen)")
    args = parser.parse_args()

    settings = get_settings()

    if args.loop:
        run_loop(settings)
    else:
        run_once(settings)
