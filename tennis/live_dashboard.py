"""
live_dashboard.py — Generate a static HTML dashboard from bet_log_*.csv files.

Run standalone:
    python tennis/live_dashboard.py --tours atp wta
Or called automatically from predict_today.py / scheduler.py.
"""

import os
import sys
import argparse
import pandas as pd

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
OUT_FILE   = os.path.join(DATA_DIR, "live_dashboard.html")


def _log_file(tour: str) -> str:
    return os.path.join(DATA_DIR, f"bet_log_{tour}.csv")


def _colour(val: float) -> str:
    return "#2ecc71" if val >= 0 else "#e74c3c"


def _load(tour: str) -> pd.DataFrame:
    path = _log_file(tour)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["pnl"]   = pd.to_numeric(df["pnl"],   errors="coerce")
    df["stake"] = pd.to_numeric(df["stake"], errors="coerce")
    df["odds"]  = pd.to_numeric(df["odds"],  errors="coerce")
    return df


def _summary_cards(df: pd.DataFrame) -> str:
    settled = df[df["result"].astype(str).str.strip().str.upper().isin(["W", "L"])].copy()
    total   = len(df)
    pending = total - len(settled)
    wins    = int((settled["result"].str.upper() == "W").sum()) if not settled.empty else 0
    n_s     = len(settled)
    win_pct = wins / n_s * 100 if n_s else 0
    staked  = settled["stake"].sum() if not settled.empty else 0
    pnl     = settled["pnl"].sum()   if not settled.empty else 0
    roi     = pnl / staked * 100     if staked else 0

    def card(label, value, colour=None):
        style = f"color:{colour};" if colour else ""
        return f"<div class='card'><div class='card-label'>{label}</div><div class='card-value' style='{style}'>{value}</div></div>"

    return f"""
    <div class='cards'>
      {card("Total Bets", total)}
      {card("Settled", n_s)}
      {card("Pending", pending)}
      {card("Win Rate", f"{win_pct:.1f}%", _colour(win_pct - 50))}
      {card("ROI", f"{roi:+.1f}%", _colour(roi))}
      {card("P&amp;L", f"£{pnl:+.2f}", _colour(pnl))}
      {card("Total Staked", f"£{staked:.2f}")}
    </div>"""


def _bankroll_chart(df: pd.DataFrame, chart_id: str, starting_bankroll: float = 100.0) -> str:
    settled = df[df["result"].astype(str).str.strip().str.upper().isin(["W", "L"])].copy()
    if settled.empty:
        return "<p style='color:#888'>No settled bets yet — bankroll chart will appear here.</p>"
    settled = settled.sort_values("date")
    bankroll = starting_bankroll
    dates, values = [], []
    for _, row in settled.iterrows():
        bankroll += float(row["pnl"]) if pd.notna(row["pnl"]) else 0
        dates.append(str(row["date"])[:10])
        values.append(round(bankroll, 2))

    labels_js = str(dates).replace("'", '"')
    vals_js   = str(values)
    return f"""
    <canvas id="{chart_id}" style="max-height:280px"></canvas>
    <script>
    new Chart(document.getElementById('{chart_id}'), {{
      type: 'line',
      data: {{
        labels: {labels_js},
        datasets: [{{
          label: 'Bankroll',
          data: {vals_js},
          borderColor: '#f39c12',
          backgroundColor: 'rgba(243,156,18,0.08)',
          borderWidth: 2, pointRadius: 2, tension: 0.3, fill: true
        }}]
      }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ ticks: {{ maxTicksLimit: 10, color: '#aaa' }}, grid: {{ color: '#222' }} }},
          y: {{ title: {{ display: true, text: 'Bankroll (£)', color: '#aaa' }},
               ticks: {{ color: '#aaa' }}, grid: {{ color: '#222' }} }}
        }}
      }}
    }});
    </script>"""


def _bet_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p style='color:#888'>No bets logged yet.</p>"
    rows = ""
    for _, r in df.sort_values("date", ascending=False).iterrows():
        res = str(r.get("result", "")).strip().upper()
        if res == "W":
            res_html = "<span style='color:#2ecc71;font-weight:bold'>W</span>"
        elif res == "L":
            res_html = "<span style='color:#e74c3c;font-weight:bold'>L</span>"
        else:
            res_html = "<span style='color:#888'>—</span>"

        pnl_val = r.get("pnl", "")
        if pd.notna(pnl_val) and str(pnl_val).strip() != "":
            pnl_html = f"<span style='color:{_colour(float(pnl_val))}'>£{float(pnl_val):+.2f}</span>"
        else:
            pnl_html = "<span style='color:#888'>—</span>"

        stake_val = r.get("stake", "")
        stake_str = f"£{float(stake_val):.2f}" if pd.notna(stake_val) and str(stake_val).strip() != "" else "—"

        rows += f"""<tr>
          <td>{str(r['date'])[:10]}</td>
          <td>{r.get('tour','')}</td>
          <td>{r['home']} vs {r['away']}</td>
          <td>{r.get('surface','')}</td>
          <td><strong>{r['bet_side']}</strong></td>
          <td>{r['line']}</td>
          <td>{r['odds']}</td>
          <td>{r.get('edge','')}%</td>
          <td>{float(r.get('model_prob',0))*100:.1f}%</td>
          <td>{stake_str}</td>
          <td>{res_html}</td>
          <td>{pnl_html}</td>
        </tr>"""

    return f"""
    <div style="overflow-x:auto">
    <table class='bet-table'>
      <thead><tr>
        <th>Date</th><th>Tour</th><th>Match</th><th>Surface</th>
        <th>Side</th><th>Line</th><th>Odds</th><th>Edge</th>
        <th>Model%</th><th>Stake</th><th>Result</th><th>P&amp;L</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
    </div>"""


def _yearly_table(df: pd.DataFrame) -> str:
    settled = df[df["result"].astype(str).str.strip().str.upper().isin(["W", "L"])].copy()
    if settled.empty:
        return "<p style='color:#888'>No settled bets yet.</p>"
    settled["year"] = settled["date"].dt.year
    rows = ""
    for yr, grp in settled.groupby("year"):
        wins  = int((grp["result"].str.upper() == "W").sum())
        n     = len(grp)
        staked = grp["stake"].sum()
        pnl   = grp["pnl"].sum()
        roi   = pnl / staked * 100 if staked else 0
        rows += f"""<tr>
          <td>{yr}</td><td>{n}</td><td>{wins}</td>
          <td>{wins/n*100:.1f}%</td>
          <td style='color:{_colour(pnl)}'>£{pnl:+.2f}</td>
          <td style='color:{_colour(roi)}'>{roi:+.1f}%</td>
        </tr>"""
    return f"""
    <table class='bet-table'>
      <thead><tr><th>Year</th><th>Bets</th><th>Won</th><th>Win%</th><th>P&amp;L</th><th>ROI</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def _filter_breakdown(df: pd.DataFrame) -> str:
    settled = df[df["result"].astype(str).str.strip().str.upper().isin(["W", "L"])].copy()
    if settled.empty:
        return "<p style='color:#888'>No settled bets yet.</p>"
    rows = ""
    for side in ["Over", "Under"]:
        grp = settled[settled["bet_side"] == side]
        if grp.empty:
            continue
        wins  = int((grp["result"].str.upper() == "W").sum())
        n     = len(grp)
        staked = grp["stake"].sum()
        pnl   = grp["pnl"].sum()
        roi   = pnl / staked * 100 if staked else 0
        rows += f"""<tr>
          <td><strong>{side}</strong></td><td>{n}</td><td>{wins}</td>
          <td>{wins/n*100:.1f}%</td>
          <td style='color:{_colour(pnl)}'>£{pnl:+.2f}</td>
          <td style='color:{_colour(roi)}'>{roi:+.1f}%</td>
        </tr>"""
    return f"""
    <table class='bet-table'>
      <thead><tr><th>Side</th><th>Bets</th><th>Won</th><th>Win%</th><th>P&amp;L</th><th>ROI</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def _tour_section(tour: str, chart_idx: int) -> str:
    df = _load(tour)
    if df.empty:
        return f"<section><h2>{tour.upper()}</h2><p style='color:#888'>No bet log found.</p></section>"

    return f"""
    <section>
      <h2>{tour.upper()} — Live Bet Tracker</h2>
      <h3>Summary</h3>
      {_summary_cards(df)}
      <h3>Bankroll Curve</h3>
      {_bankroll_chart(df, f"chart_{chart_idx}")}
      <h3>By Side</h3>
      {_filter_breakdown(df)}
      <h3>Yearly Breakdown</h3>
      {_yearly_table(df)}
      <h3>All Bets</h3>
      {_bet_table(df)}
    </section>"""


def generate_dashboard(tours: list[str]) -> str | None:
    os.makedirs(DATA_DIR, exist_ok=True)
    sections = "".join(_tour_section(t, i) for i, t in enumerate(tours))
    today = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tennis Bet Tracker — Live Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0f1117; color: #e0e0e0; margin: 0; padding: 20px; }}
  h1   {{ color: #f39c12; text-align: center; margin-bottom: 4px; }}
  h2   {{ color: #3498db; border-bottom: 1px solid #2c3e50; padding-bottom: 8px; margin-top: 0; }}
  h3   {{ color: #f39c12; margin-top: 20px; font-size: 0.95em; }}
  .subtitle {{ text-align: center; color: #aaa; margin-bottom: 30px; font-size: 0.9em; }}
  section {{ background: #1a1d26; border-radius: 12px; padding: 24px; margin-bottom: 30px; box-shadow: 0 4px 20px rgba(0,0,0,.4); }}
  .cards {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 12px 0; }}
  .card {{ background: #252836; border-radius: 8px; padding: 14px 20px; min-width: 120px; flex: 1; text-align: center; }}
  .card-label {{ color: #aaa; font-size: 0.78em; text-transform: uppercase; letter-spacing: 0.05em; }}
  .card-value {{ font-size: 1.4em; font-weight: bold; margin-top: 4px; }}
  table.bet-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.85em; }}
  table.bet-table th {{ background: #2c3e50; color: #ecf0f1; padding: 8px 10px; text-align: left; white-space: nowrap; }}
  table.bet-table td {{ padding: 7px 10px; border-bottom: 1px solid #2a2d3a; white-space: nowrap; }}
  table.bet-table tr:hover td {{ background: rgba(52,152,219,.07); }}
  canvas {{ margin-top: 12px; }}
  p {{ color: #aaa; }}
</style>
</head>
<body>
<h1>Tennis Bet Tracker</h1>
<p class="subtitle">Live Dashboard &nbsp;|&nbsp; Last updated: {today}</p>
{sections}
</body>
</html>"""

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard saved: {OUT_FILE}")
    return OUT_FILE


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tours", nargs="+", default=["atp"], choices=["atp", "wta"])
    args = parser.parse_args()
    generate_dashboard(args.tours)
