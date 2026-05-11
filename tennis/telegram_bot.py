"""
telegram_bot.py — Send daily predictions and heartbeat to Telegram.
Set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID as environment variables.

Interactive mode:
    python tennis/telegram_bot.py
"""

import os
import requests


TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def send(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("  [Telegram] Not configured — skipping.")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        r = requests.post(url, json={
            "chat_id":    TELEGRAM_CHAT_ID,
            "text":       message,
            "parse_mode": "HTML",
        }, timeout=10)
        if r.status_code == 200:
            print("  [Telegram] Message sent.")
        else:
            print(f"  [Telegram] Failed: {r.text}")
    except Exception as e:
        print(f"  [Telegram] Error: {e}")


def send_predictions(all_matches: list, bets: list, surface: str, date: str,
                     bankroll: float, kelly_fraction: float = 0.25, max_stake: float = 0.05):
    if not all_matches:
        send(
            f"🎾 <b>ATP Tennis — {date}</b>\n\n"
            f"Surface: {surface}\n"
            f"No matches found today.\n\n"
            f"✅ Daily heartbeat — system running."
        )
        return

    lines = [
        f"🎾 <b>ATP Tennis — {date}</b>",
        f"Surface: {surface} | Matches: {len(all_matches)}\n",
        f"<b>📋 All Today's Matches:</b>",
    ]

    # All matches section
    for i, (home, away, line, over_odds, under_odds, bet, stake, time_uk) in enumerate(all_matches, 1):
        flag = "⚡️" if bet else "—"
        lines.append(f"{i}. {flag} {home} vs {away}  |  {time_uk}  |  O/U {line}  |  {over_odds}/{under_odds}")

    # Bets section
    if bets:
        total_stake = sum(s for _, _, _, s in bets if s)
        lines.append(f"\n<b>✅ Recommended Bets ({len(bets)}):</b>")
        for home, away, bet, stake in bets:
            lines.append(
                f"⚡️ <b>{home} vs {away}</b>\n"
                f"   {bet['side']} {bet['line']} games @ {bet['odds']}\n"
                f"   Edge: {bet['edge']}% | Model: {bet['model_prob']*100:.1f}%\n"
                f"   Stake: £{stake:.2f}"
            )
        lines.append(
            f"\n💰 Bankroll: £{bankroll:.2f} | Kelly: {kelly_fraction*100:.0f}% | "
            f"Total staked: £{total_stake:.2f} ({total_stake/bankroll*100:.1f}%)"
        )
    else:
        lines.append("\n<b>No bets qualify today</b> (no match passed 4% edge threshold).")

    lines.append("\n✅ Daily heartbeat — system running.")
    send("\n".join(lines))


def send_pnl_update(summary: dict, date: str):
    if not summary or summary.get("total_bets", 0) == 0:
        return
    msg = (
        f"📊 <b>P&L Update — {date}</b>\n\n"
        f"Settled bets : {summary['total_bets']}\n"
        f"Won          : {summary['wins']} ({summary['win_rate']:.1f}%)\n"
        f"Total staked : £{summary['total_staked']:.2f}\n"
        f"Total P&L    : £{summary['total_pnl']:+.2f}\n"
        f"ROI          : {summary['roi']:+.1f}%"
    )
    send(msg)


def send_heartbeat(date: str, status: str = "OK"):
    send(f"💓 Heartbeat — {date} — Status: {status}")


def _ask(prompt: str, default: str) -> str:
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else default


def interactive_send():
    """Ask questions then send a custom Telegram message with Kelly stake info."""
    print("\n=== Telegram Message Generator ===\n")

    bankroll = float(_ask("Your bankroll (£)", "100"))

    print("Kelly fraction options:")
    print("  0.10 = 10% Kelly (very safe)")
    print("  0.25 = Quarter Kelly (recommended)")
    print("  0.50 = Half Kelly (aggressive)")
    kelly = float(_ask("Kelly fraction", "0.25"))

    max_stake = float(_ask("Max stake per bet (as decimal, e.g. 0.05 = 5%)", "0.05"))

    print("\nEnter your bets (leave match blank to finish):")
    bets = []
    while True:
        match = input("  Match (e.g. Sinner vs Alcaraz) or Enter to finish: ").strip()
        if not match:
            break
        side   = input("  Side (Over/Under): ").strip()
        line   = input("  Line (e.g. 22.5): ").strip()
        odds   = input("  Odds (e.g. 1.85): ").strip()
        edge   = input("  Edge % (e.g. 6.2): ").strip()
        model  = input("  Model prob % (e.g. 63.4): ").strip()
        kelly_raw = input("  Full Kelly % from model (e.g. 18.5): ").strip()

        try:
            kelly_full = float(kelly_raw) / 100
            stake = bankroll * min(kelly_full * kelly, max_stake)
            bets.append({
                "match":  match,
                "side":   side,
                "line":   line,
                "odds":   odds,
                "edge":   edge,
                "model":  model,
                "stake":  round(stake, 2),
            })
            print(f"  → Stake: £{stake:.2f}\n")
        except ValueError:
            print("  Invalid input, skipping this bet.\n")

    if not bets:
        print("No bets entered.")
        return

    import pandas as pd
    date = pd.Timestamp.now().strftime("%Y-%m-%d")
    total_stake = sum(b["stake"] for b in bets)

    lines = [
        f"🎾 <b>ATP Tennis Predictions — {date}</b>",
        f"Bankroll: £{bankroll:.2f} | Kelly: {kelly*100:.0f}% | Max stake: {max_stake*100:.0f}%\n",
    ]
    for b in bets:
        lines.append(
            f"⚡ <b>{b['match']}</b>\n"
            f"   {b['side']} {b['line']} games @ {b['odds']}\n"
            f"   Edge: {b['edge']}% | Model: {b['model']}%\n"
            f"   Stake: £{b['stake']:.2f}"
        )
    lines.append(f"\n💰 Total stake: £{total_stake:.2f} ({total_stake/bankroll*100:.1f}% of bankroll)")
    lines.append("\n✅ Daily heartbeat — system running.")

    message = "\n".join(lines)
    print("\n--- Preview ---")
    print(message)
    print("---------------")

    confirm = input("\nSend this to Telegram? (y/n): ").strip().lower()
    if confirm == "y":
        send(message)
    else:
        print("Cancelled.")


if __name__ == "__main__":
    interactive_send()
