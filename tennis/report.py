import os
import pandas as pd

KELLY_FRACTIONS = [
    ("Full Kelly",    1.00),
    ("Half Kelly",    0.50),
    ("Quarter Kelly", 0.25),
]


def _simulate_bankroll(bets_df, starting_bankroll, kelly_fraction, max_stake_pct):
    bankroll = starting_bankroll
    records = []
    for _, row in bets_df.iterrows():
        raw_kelly = float(row["kelly"])
        stake = bankroll * min(raw_kelly * kelly_fraction, max_stake_pct)
        stake = max(0.0, stake)
        won = bool(row["won"])
        pnl = stake * (float(row["bk_odds"]) - 1.0) if won else -stake
        bankroll += pnl
        records.append({"date": row["date"], "bankroll": bankroll, "pnl": pnl, "won": won})
    return pd.DataFrame(records)


def _summary(bets_df, starting_bankroll, kelly_fraction, max_stake_pct):
    if bets_df.empty:
        return {"bets": 0, "wins": 0, "win_pct": 0, "roi": 0,
                "final_bankroll": starting_bankroll, "max_dd": 0, "total_pnl": 0}
    sim = _simulate_bankroll(bets_df, starting_bankroll, kelly_fraction, max_stake_pct)
    total_pnl = sim["pnl"].sum()
    final_br  = sim["bankroll"].iloc[-1]
    peak      = sim["bankroll"].cummax()
    max_dd    = ((peak - sim["bankroll"]) / peak).max() * 100
    wins      = int(bets_df["won"].sum())
    n         = len(bets_df)
    return {
        "bets": n, "wins": wins, "win_pct": wins / n * 100,
        "roi": total_pnl / starting_bankroll * 100,
        "final_bankroll": final_br, "max_dd": max_dd,
        "total_pnl": total_pnl, "curve": sim,
    }


def _yearly_roi(bets_df, starting_bankroll, kelly_fraction, max_stake_pct):
    if bets_df.empty:
        return {}
    df = bets_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    rows = {}
    for yr, grp in df.groupby("year"):
        sim = _simulate_bankroll(grp, starting_bankroll, kelly_fraction, max_stake_pct)
        pnl = sim["pnl"].sum()
        w   = int(grp["won"].sum())
        n   = len(grp)
        rows[yr] = {"bets": n, "wins": w, "win_pct": w / n * 100,
                    "pnl": pnl, "roi": pnl / starting_bankroll * 100}
    return rows


def _colour(val):
    return "#2ecc71" if val >= 0 else "#e74c3c"


def _kelly_comparison_table(stats_by_kelly):
    rows_html = ""
    for label, kf in KELLY_FRACTIONS:
        s = stats_by_kelly[label]
        rows_html += f"""
        <tr>
          <td><strong>{label}</strong></td>
          <td>{kf:.2f}</td>
          <td>{s['bets']:,}</td>
          <td>{s['win_pct']:.1f}%</td>
          <td style="color:{_colour(s['roi'])};font-weight:bold">{s['roi']:+.1f}%</td>
          <td style="color:{_colour(s['total_pnl'])};font-weight:bold">{s['total_pnl']:+.2f}</td>
          <td>{s['final_bankroll']:.2f}</td>
          <td>{s['max_dd']:.1f}%</td>
        </tr>"""
    return f"""
    <table class="summary-table">
      <thead>
        <tr>
          <th>Strategy</th><th>Fraction</th><th>Bets</th><th>Win %</th>
          <th>ROI</th><th>P&amp;L</th><th>Final Bankroll</th><th>Max Drawdown</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>"""


def _bankroll_chart(stats_by_kelly, chart_id):
    datasets = []
    colours  = ["#f39c12", "#3498db", "#2ecc71"]
    for (label, _kf), col in zip(KELLY_FRACTIONS, colours):
        curve = stats_by_kelly[label].get("curve")
        if curve is None or curve.empty:
            continue
        dates  = curve["date"].astype(str).tolist()
        values = [round(v, 2) for v in curve["bankroll"].tolist()]
        datasets.append({"label": label, "colour": col, "dates": dates, "values": values})
    if not datasets:
        return ""
    labels_js = str(datasets[0]["dates"]).replace("'", '"')
    ds_js_parts = []
    for d in datasets:
        vals = str(d["values"])
        ds_js_parts.append(
            f"{{label:'{d['label']}',data:{vals},"
            f"borderColor:'{d['colour']}',backgroundColor:'transparent',"
            f"borderWidth:2,pointRadius:0,tension:0.3}}"
        )
    ds_js = ",".join(ds_js_parts)
    return f"""
    <canvas id="{chart_id}" style="max-height:320px"></canvas>
    <script>
    new Chart(document.getElementById('{chart_id}'), {{
      type: 'line',
      data: {{
        labels: {labels_js},
        datasets: [{ds_js}]
      }},
      options: {{
        responsive: true,
        interaction: {{mode:'index',intersect:false}},
        plugins: {{legend: {{display:true}}}},
        scales: {{
          x: {{ticks:{{maxTicksLimit:12}}}},
          y: {{title:{{display:true,text:'Bankroll'}}}}
        }}
      }}
    }});
    </script>"""


def _yearly_table(yearly_data):
    if not yearly_data:
        return "<p>No yearly data.</p>"
    years  = sorted(set(yr for d in yearly_data.values() for yr in d))
    header = "<tr><th>Year</th><th>Bets</th>" + "".join(f"<th colspan='3'>{lbl}</th>" for lbl, _ in KELLY_FRACTIONS) + "</tr>"
    sub    = "<tr><th></th><th></th>" + "".join("<th>W%</th><th>P&amp;L</th><th>ROI</th>" for _ in KELLY_FRACTIONS) + "</tr>"
    body   = ""
    first_label = KELLY_FRACTIONS[0][0]
    for yr in years:
        first_yd = yearly_data.get(first_label, {}).get(yr)
        bets_cell = f"<td>{first_yd['bets']:,}</td>" if first_yd else "<td>-</td>"
        body += f"<tr><td>{yr}</td>{bets_cell}"
        for label, _ in KELLY_FRACTIONS:
            yd = yearly_data.get(label, {}).get(yr)
            if yd:
                body += (f"<td>{yd['win_pct']:.1f}%</td>"
                         f"<td style='color:{_colour(yd['pnl'])}'>{yd['pnl']:+.1f}</td>"
                         f"<td style='color:{_colour(yd['roi'])}'>{yd['roi']:+.1f}%</td>")
            else:
                body += "<td>-</td><td>-</td><td>-</td>"
        body += "</tr>"
    return f"<table class='summary-table'><thead>{header}{sub}</thead><tbody>{body}</tbody></table>"


def _section_html(filter_label, bets_df, starting_bankroll, max_stake_pct, chart_id):
    stats_by_kelly  = {}
    yearly_by_kelly = {}
    for label, kf in KELLY_FRACTIONS:
        stats_by_kelly[label]  = _summary(bets_df, starting_bankroll, kf, max_stake_pct)
        yearly_by_kelly[label] = _yearly_roi(bets_df, starting_bankroll, kf, max_stake_pct)

    real_count = int(bets_df["real_odds"].sum()) if "real_odds" in bets_df.columns else 0
    sim_count  = len(bets_df) - real_count
    odds_note  = (f"<small>Real odds: {real_count} bets &nbsp;|&nbsp; Simulated odds: {sim_count} bets</small>"
                  if real_count > 0 else "")

    return f"""
    <section>
      <h2>{filter_label} Bets</h2>
      {odds_note}
      <h3>Kelly Fraction Comparison</h3>
      {_kelly_comparison_table(stats_by_kelly)}
      <h3>Bankroll Growth</h3>
      {_bankroll_chart(stats_by_kelly, chart_id)}
      <h3>Yearly Breakdown</h3>
      {_yearly_table(yearly_by_kelly)}
    </section>"""


def generate_html(tour, starting_bankroll, kelly_fraction, max_stake, data_dir, suffix="", market="totals"):
    filters = [
        ("over",       "Over Sets" if market == "sets" else "Over Games"),
        ("under",      "Under Sets" if market == "sets" else "Under Games"),
        ("both",       "Over + Under Sets" if market == "sets" else "Over + Under Games"),
        ("under_opt",  "Under Optimised (bo5 + gap>0)"),
    ]
    sections = []
    chart_idx = 0
    for filt, label in filters:
        csv_path = os.path.join(data_dir, f"backtest_results_{tour}_{market}_{filt}{suffix}.csv")
        if not os.path.exists(csv_path):
            # fallback to old filename pattern for backwards compatibility
            csv_path = os.path.join(data_dir, f"backtest_results_{tour}_{filt}{suffix}.csv")
            if not os.path.exists(csv_path):
                continue
        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"])
        if "real_odds" not in df.columns:
            df["real_odds"] = False
        sections.append(_section_html(label, df, starting_bankroll, max_stake, f"chart_{chart_idx}"))
        chart_idx += 1

    if not sections:
        print("No CSV data found — run the backtest first.")
        return None

    tour_upper   = tour.upper()
    market_label = "Total Sets" if market == "sets" else "Total Games"
    odds_type    = "Real Betfair Odds" if suffix == "_real" else "Simulated Bookmaker Odds"
    fracs_desc   = " | ".join(f"{l} ({f:.2f})" for l, f in KELLY_FRACTIONS)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Barnett-Clarke Tennis Model — {tour_upper} {market_label}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body {{ font-family: Arial, sans-serif; background: #0f1117; color: #eee; margin: 0; padding: 20px; }}
  h1   {{ color: #f39c12; text-align: center; }}
  h2   {{ color: #3498db; border-bottom: 1px solid #333; padding-bottom: 6px; }}
  h3   {{ color: #bbb; font-size: 1em; }}
  .subtitle {{ text-align: center; color: #aaa; margin-bottom: 30px; }}
  section {{ background: #1a1d27; border-radius: 8px; padding: 20px; margin-bottom: 30px; }}
  table.summary-table {{ width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 0.9em; }}
  table.summary-table th {{ background: #252836; color: #f39c12; padding: 8px; text-align: center; }}
  table.summary-table td {{ padding: 7px 10px; text-align: center; border-bottom: 1px solid #2a2d3a; }}
  table.summary-table tr:hover td {{ background: #22253a; }}
  small {{ color: #888; display: block; margin-bottom: 8px; }}
  canvas {{ margin-top: 10px; }}
</style>
</head>
<body>
<h1>Barnett-Clarke Tennis {market_label} Model &mdash; {tour_upper}</h1>
<p class="subtitle">{odds_type} &nbsp;|&nbsp; Starting Bankroll: {starting_bankroll:.0f} &nbsp;|&nbsp; {fracs_desc}</p>
{''.join(sections)}
</body>
</html>"""

    out_path = os.path.join(data_dir, f"backtest_report_{tour}_{market}{suffix}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report saved: {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--tour",      default="atp")
    p.add_argument("--bankroll",  type=float, default=1000.0)
    p.add_argument("--kelly",     type=float, default=0.25)
    p.add_argument("--max-stake", type=float, default=0.05)
    p.add_argument("--data-dir",  default="tennis/data")
    p.add_argument("--suffix",    default="")
    p.add_argument("--market",    default="totals", choices=["totals", "sets"])
    args = p.parse_args()
    generate_html(args.tour, args.bankroll, args.kelly, args.max_stake, args.data_dir, args.suffix, market=args.market)
