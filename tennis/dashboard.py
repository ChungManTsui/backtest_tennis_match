"""
dashboard.py — Live Streamlit dashboard for ATP/WTA bet tracking.
Run: streamlit run tennis/dashboard.py --server.port 8501
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st
from datetime import date

DATA_DIR       = os.path.join(os.path.dirname(__file__), "data")
BANKROLL_FILE  = os.path.join(DATA_DIR, "bankroll.json")

st.set_page_config(
    page_title="Tennis Bet Tracker",
    page_icon="🎾",
    layout="wide",
)

st.title("🎾 Barnett-Clarke Tennis Model — Live Dashboard")
st.caption(f"Live tracking dashboard — {date.today()} | Auto-refreshes every 60s")


def load_bankroll() -> float:
    if os.path.exists(BANKROLL_FILE):
        with open(BANKROLL_FILE) as f:
            return float(json.load(f).get("bankroll", 100.0))
    return 100.0


def load_strategies() -> list[dict]:
    strategies = []
    if not os.path.exists(DATA_DIR):
        return strategies
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.startswith("strategy_") or not fname.endswith(".json"):
            continue
        path = os.path.join(DATA_DIR, fname)
        try:
            with open(path) as f:
                strategies.append(json.load(f))
        except Exception:
            continue
    return strategies


def render_strategy_card():
    strategies = load_strategies()
    st.subheader("⚙️ Running Strategy")
    if not strategies:
        st.info("No strategy info yet — start the scheduler to populate this.")
        st.divider()
        return

    filter_labels = {
        "both":      "Over + Under",
        "over":      "Over Only",
        "under":     "Under Only",
        "under_opt": "Under Opt (bo5 GS)",
    }
    market_labels = {
        "totals": "Total Games",
        "sets":   "Total Sets",
    }

    for s in strategies:
        tours    = ", ".join(t.upper() for t in s.get("tours", []))
        market   = market_labels.get(s.get("market", "totals"), s.get("market", "totals"))
        filt     = filter_labels.get(s.get("filter", "both"), s.get("filter", "both"))
        kelly    = s.get("kelly_fraction", 0.25)
        stake    = s.get("max_stake", 0.05)
        bankroll = s.get("bankroll", 100.0)
        last     = s.get("last_run", "—")
        next_r   = s.get("next_run", "—")

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Tours",     tours)
        c2.metric("Market",    market)
        c3.metric("Filter",    filt)
        c4.metric("Kelly",     f"{kelly*100:.0f}%")
        c5.metric("Max Stake", f"{stake*100:.0f}%")
        c6.metric("Bankroll",  f"£{bankroll:.2f}")
        st.caption(f"Last run: {last}  |  Next run: {next_r}")

    st.divider()


@st.cache_data(ttl=60)
def load_log(tour: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"bet_log_{tour}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"]  = pd.to_datetime(df["date"],  errors="coerce")
    df["pnl"]   = pd.to_numeric(df["pnl"],   errors="coerce")
    df["stake"] = pd.to_numeric(df["stake"], errors="coerce")
    df["odds"]  = pd.to_numeric(df["odds"],  errors="coerce")
    df["edge"]  = pd.to_numeric(df["edge"],  errors="coerce")
    return df


def colour_result(val):
    if str(val).upper() == "W":
        return "background-color:#1a3a2a;color:#2ecc71"
    elif str(val).upper() == "L":
        return "background-color:#3a1a1a;color:#e74c3c"
    return ""


def render_tour(tour: str):
    df = load_log(tour)

    st.header(f"{tour.upper()} — Bet Tracker")

    if df.empty:
        st.warning(f"No bets logged yet for {tour.upper()}. Run scheduler.py to start tracking.")
        return

    settled = df[df["result"].astype(str).str.strip().str.upper().isin(["W", "L"])].copy()
    pending = df[~df["result"].astype(str).str.strip().str.upper().isin(["W", "L"])].copy()

    # ── Summary metrics ───────────────────────────────────────────────────────
    total_bets   = len(settled)
    wins         = int((settled["result"].str.upper() == "W").sum()) if not settled.empty else 0
    total_staked = settled["stake"].sum() if not settled.empty else 0
    total_pnl    = settled["pnl"].sum()   if not settled.empty else 0
    roi          = total_pnl / total_staked * 100 if total_staked > 0 else 0
    win_rate     = wins / total_bets * 100 if total_bets > 0 else 0
    bankroll     = load_bankroll()

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Bankroll",     f"£{bankroll:.2f}")
    c2.metric("Settled Bets", total_bets)
    c3.metric("Pending",      len(pending))
    c4.metric("Win Rate",     f"{win_rate:.1f}%",  delta=f"{win_rate-50:+.1f}% vs 50%")
    c5.metric("Total Staked", f"£{total_staked:.2f}")
    c6.metric("Total P&L",    f"£{total_pnl:+.2f}", delta=f"{roi:+.1f}% ROI",
              delta_color="normal")
    c7.metric("Total Bets",   len(df))

    st.divider()

    # ── Bankroll curve ────────────────────────────────────────────────────────
    if not settled.empty:
        st.subheader("📈 Bankroll Curve")
        s = settled.sort_values("date").copy()
        s["bankroll"] = bankroll - total_pnl + s["pnl"].cumsum()
        st.line_chart(s[["date", "bankroll"]].set_index("date"))
        st.divider()

    # ── By surface / side ─────────────────────────────────────────────────────
    if not settled.empty:
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.subheader("By Surface")
            rows = []
            for surf in settled["surface"].dropna().unique():
                g = settled[settled["surface"] == surf]
                w = int((g["result"].str.upper() == "W").sum())
                pnl = g["pnl"].sum()
                rows.append({"Surface": surf, "Bets": len(g), "Won": w,
                             "Win%": f"{w/len(g)*100:.1f}%", "P&L": f"£{pnl:+.2f}"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        with col_b:
            st.subheader("By Side")
            rows = []
            for side in ["Over", "Under"]:
                g = settled[settled["bet_side"] == side]
                if g.empty:
                    continue
                w = int((g["result"].str.upper() == "W").sum())
                pnl = g["pnl"].sum()
                rows.append({"Side": side, "Bets": len(g), "Won": w,
                             "Win%": f"{w/len(g)*100:.1f}%", "P&L": f"£{pnl:+.2f}"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        with col_c:
            st.subheader("By Year")
            settled["year"] = settled["date"].dt.year
            rows = []
            for yr, g in settled.groupby("year"):
                w = int((g["result"].str.upper() == "W").sum())
                pnl = g["pnl"].sum()
                stk = g["stake"].sum()
                rows.append({"Year": yr, "Bets": len(g), "Won": w,
                             "Win%": f"{w/len(g)*100:.1f}%",
                             "P&L": f"£{pnl:+.2f}",
                             "ROI": f"{pnl/stk*100:+.1f}%" if stk else "—"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        st.divider()

    # ── Pending bets ──────────────────────────────────────────────────────────
    if not pending.empty:
        st.subheader(f"⏳ Pending Bets ({len(pending)})")
        p = pending.copy()
        p["date"]  = p["date"].dt.strftime("%Y-%m-%d")
        p["match"] = p["home"] + " vs " + p["away"]
        p["bet"]   = p["bet_side"] + " " + p["line"].astype(str) + " @ " + p["odds"].astype(str)
        p["stake"] = p["stake"].apply(lambda x: f"£{x:.2f}" if pd.notna(x) else "—")
        st.dataframe(
            p[["date", "match", "surface", "bet", "edge", "stake"]].rename(
                columns={"date": "Date", "match": "Match", "surface": "Surface",
                         "bet": "Bet", "edge": "Edge%", "stake": "Stake"}),
            hide_index=True, use_container_width=True)
        st.caption("Results are auto-filled each morning by the scheduler.")
        st.divider()

    # ── All settled bets ──────────────────────────────────────────────────────
    st.subheader("📋 All Settled Bets")
    if settled.empty:
        st.info("No settled bets yet.")
    else:
        s = settled.sort_values("date", ascending=False).copy()
        s["date"]  = s["date"].dt.strftime("%Y-%m-%d")
        s["match"] = s["home"] + " vs " + s["away"]
        s["bet"]   = s["bet_side"] + " " + s["line"].astype(str) + " @ " + s["odds"].astype(str)
        s["stake"] = s["stake"].apply(lambda x: f"£{x:.2f}" if pd.notna(x) else "—")
        s["P&L"]   = s["pnl"].apply(lambda x: f"£{x:+.2f}" if pd.notna(x) else "—")
        show = s[["date", "match", "surface", "bet", "edge", "stake", "result", "P&L"]].rename(
            columns={"date": "Date", "match": "Match", "surface": "Surface",
                     "bet": "Bet", "edge": "Edge%", "stake": "Stake", "result": "Result"})
        st.dataframe(
            show.style.map(colour_result, subset=["Result"]),
            hide_index=True, use_container_width=True)


# ── Strategy card ─────────────────────────────────────────────────────────────
render_strategy_card()

# ── Tour tabs ─────────────────────────────────────────────────────────────────
atp_exists = os.path.exists(os.path.join(DATA_DIR, "bet_log_atp.csv"))
wta_exists = os.path.exists(os.path.join(DATA_DIR, "bet_log_wta.csv"))

if atp_exists and wta_exists:
    tab_atp, tab_wta = st.tabs(["ATP", "WTA"])
    with tab_atp:
        render_tour("atp")
    with tab_wta:
        render_tour("wta")
elif atp_exists:
    render_tour("atp")
elif wta_exists:
    render_tour("wta")
else:
    st.warning("No bet logs found. Run the scheduler first.")

st.divider()
st.caption("Barnett-Clarke Markov Chain model | Bet365/Pinnacle odds via The Odds API | Auto-updated daily")
