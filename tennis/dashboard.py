"""
dashboard.py — Live Streamlit dashboard.
Run: streamlit run tennis/dashboard.py
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
from datetime import date

import json

LOG_FILE      = "tennis/data/bet_log.csv"
BANKROLL_FILE = "tennis/data/bankroll.json"


def load_initial_bankroll() -> float:
    if os.path.exists(BANKROLL_FILE):
        with open(BANKROLL_FILE) as f:
            return float(json.load(f)["bankroll"])
    return 100.0

st.set_page_config(page_title="ATP Tennis Model", page_icon="🎾", layout="wide")

st.title("🎾 Barnett-Clarke ATP Total Games Model")
st.caption(f"Live tracking dashboard — updated daily | {date.today()}")


@st.cache_data(ttl=60)
def load_log():
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()
    df = pd.read_csv(LOG_FILE)
    df["date"]  = pd.to_datetime(df["date"], errors="coerce")
    df["pnl"]   = pd.to_numeric(df["pnl"],   errors="coerce")
    df["stake"] = pd.to_numeric(df["stake"], errors="coerce")
    df["odds"]  = pd.to_numeric(df["odds"],  errors="coerce")
    df["edge"]  = pd.to_numeric(df["edge"],  errors="coerce")
    return df


df = load_log()

if df.empty:
    st.warning("No bets logged yet. Run scheduler.py to start tracking.")
    st.stop()

settled = df[df["result"].astype(str).str.strip().str.upper().isin(["W", "L"])].copy()
pending = df[~df["result"].astype(str).str.strip().str.upper().isin(["W", "L"])].copy()

# ── Top metrics ───────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5, col6 = st.columns(6)

if not settled.empty:
    total_bets   = len(settled)
    wins         = int((settled["result"].str.upper() == "W").sum())
    total_staked = settled["stake"].sum()
    total_pnl    = settled["pnl"].sum()
    roi          = total_pnl / total_staked * 100 if total_staked > 0 else 0
    win_rate     = wins / total_bets * 100
    current_bankroll = load_initial_bankroll()

    col1.metric("Bankroll",      f"£{current_bankroll:.2f}")
    col2.metric("Total Bets",    total_bets)
    col3.metric("Win Rate",      f"{win_rate:.1f}%")
    col4.metric("Total Staked",  f"£{total_staked:.2f}")
    col5.metric("Total P&L",     f"£{total_pnl:+.2f}",
                delta=f"{roi:+.1f}% ROI",
                delta_color="normal")
    col6.metric("Pending Bets",  len(pending))
else:
    st.info("No settled bets yet — fill in W/L in bet_log.csv after matches finish.")

st.divider()

# ── Bankroll curve ────────────────────────────────────────────────────────────
if not settled.empty:
    st.subheader("📈 Bankroll Curve")
    initial_bankroll = load_initial_bankroll()
    settled_sorted = settled.sort_values("date").copy()
    settled_sorted["cumulative_pnl"] = settled_sorted["pnl"].cumsum()
    settled_sorted["bankroll"] = initial_bankroll + settled_sorted["cumulative_pnl"]

    chart_df = settled_sorted[["date", "bankroll"]].set_index("date")
    st.line_chart(chart_df)

st.divider()

# ── By surface and bet side ───────────────────────────────────────────────────
if not settled.empty:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("By Surface")
        surf_stats = []
        for surf in settled["surface"].unique():
            s = settled[settled["surface"] == surf]
            w = int((s["result"].str.upper() == "W").sum())
            pnl = s["pnl"].sum()
            surf_stats.append({
                "Surface": surf,
                "Bets": len(s),
                "Won": w,
                "Win%": f"{w/len(s)*100:.1f}%",
                "P&L": f"£{pnl:+.2f}",
            })
        st.dataframe(pd.DataFrame(surf_stats), hide_index=True)

    with col_b:
        st.subheader("By Bet Side")
        side_stats = []
        for side in ["Over", "Under"]:
            s = settled[settled["bet_side"] == side]
            if s.empty:
                continue
            w = int((s["result"].str.upper() == "W").sum())
            pnl = s["pnl"].sum()
            side_stats.append({
                "Side": side,
                "Bets": len(s),
                "Won": w,
                "Win%": f"{w/len(s)*100:.1f}%",
                "P&L": f"£{pnl:+.2f}",
            })
        st.dataframe(pd.DataFrame(side_stats), hide_index=True)

st.divider()

# ── Recent bets ───────────────────────────────────────────────────────────────
st.subheader("📋 Recent Bets")

display_df = df.sort_values("date", ascending=False).head(30).copy()
display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
display_df["match"] = display_df["home"] + " vs " + display_df["away"]
display_df["bet"] = display_df["bet_side"] + " " + display_df["line"].astype(str) + " @ " + display_df["odds"].astype(str)
display_df["stake"] = display_df["stake"].apply(lambda x: f"£{x:.2f}" if pd.notna(x) else "—")
display_df["pnl_fmt"] = display_df["pnl"].apply(lambda x: f"£{x:+.2f}" if pd.notna(x) else "—")
display_df["result"] = display_df["result"].fillna("⏳")

show = display_df[["date", "match", "surface", "bet", "edge", "stake", "result", "pnl_fmt"]]
show.columns = ["Date", "Match", "Surface", "Bet", "Edge%", "Stake", "Result", "P&L"]

def color_result(val):
    if val == "W":
        return "background-color: #d4edda; color: #155724"
    elif val == "L":
        return "background-color: #f8d7da; color: #721c24"
    return ""

st.dataframe(
    show.style.map(color_result, subset=["Result"]),
    hide_index=True,
    use_container_width=True,
)

st.divider()

# ── Pending bets ──────────────────────────────────────────────────────────────
if not pending.empty:
    st.subheader("⏳ Pending Bets (awaiting results)")
    pending_display = pending.copy()
    pending_display["date"] = pending_display["date"].dt.strftime("%Y-%m-%d")
    pending_display["match"] = pending_display["home"] + " vs " + pending_display["away"]
    pending_display["bet"] = (pending_display["bet_side"] + " " +
                               pending_display["line"].astype(str) + " @ " +
                               pending_display["odds"].astype(str))
    pending_display["stake"] = pending_display["stake"].apply(
        lambda x: f"£{x:.2f}" if pd.notna(x) else "—")
    show_p = pending_display[["date", "match", "surface", "bet", "edge", "stake"]]
    show_p.columns = ["Date", "Match", "Surface", "Bet", "Edge%", "Stake"]
    st.dataframe(show_p, hide_index=True, use_container_width=True)
    st.caption("Fill in W or L in tennis/data/bet_log.csv after matches finish.")

st.divider()
st.caption("Barnett-Clarke Markov Chain model | Bet365/Pinnacle odds via The Odds API | Auto-updated daily")
