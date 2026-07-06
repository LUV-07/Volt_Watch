"""
VoltWatch - Interactive Dashboard Builder
============================================
Produces a single self-contained HTML file (outputs/dashboard/voltwatch_dashboard.html)
with drift analysis, anomaly timeline, and model governance metrics.

This is a working, open-able-in-any-browser stand-in for the Power BI /
Tableau deliverables mentioned on the resume - those are licensed desktop
tools this environment can't run directly, but the underlying data marts
(outputs/dashboard_data/*.csv) are built exactly so they can be dropped
into either tool with zero transformation (see dashboard_data/README.md).
"""
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

OUT_DIR = "/home/claude/voltwatch/outputs"
DASH_DATA = f"{OUT_DIR}/dashboard_data"
DASH_OUT = f"{OUT_DIR}/dashboard/voltwatch_dashboard.html"

TEMPLATE = "plotly_white"
COLORS = {"bg": "#0f172a", "card": "#f8fafc", "accent": "#2563eb", "danger": "#dc2626", "ok": "#16a34a"}


def kpi_cards(eval_results):
    hybrid = eval_results["holdout_test"]["hybrid"]
    cv = eval_results["cv_isolation_forest"]["mean"]
    cards = [
        ("Hybrid ROC-AUC (holdout)", f"{hybrid['roc_auc']:.3f}"),
        ("Precision @ 90% floor", f"{hybrid['precision_90_floor']['precision']*100:.1f}%"),
        ("Recall @ 90% floor", f"{hybrid['precision_90_floor']['recall']*100:.1f}%"),
        ("F1 (best operating point)", f"{hybrid['f1_optimal']['f1']:.3f}"),
        ("IF 4-fold CV F1 (mean)", f"{cv['f1']:.3f}"),
        ("Test set size", f"{eval_results['test_set_size']:,} readings"),
    ]
    return cards


def build_drift_fig(drift: pd.DataFrame):
    fig = go.Figure()
    for meter_id, g in drift.groupby("meter_id"):
        fig.add_trace(go.Scatter(
            x=g["month"], y=g["avg_hybrid_score"], mode="lines+markers",
            name=meter_id, hovertemplate="%{x}<br>avg score: %{y:.3f}<extra>" + meter_id + "</extra>"
        ))
    fig.update_layout(
        title="Model Drift: Avg Hybrid Anomaly Score by Meter, Month over Month",
        xaxis_title="Month", yaxis_title="Avg hybrid anomaly score",
        template=TEMPLATE, height=380, legend_title="Meter",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def build_events_fig(events: pd.DataFrame):
    fig = go.Figure()
    colors = events["true_positive"].map({True: COLORS["danger"], False: "#f59e0b"})
    fig.add_trace(go.Scatter(
        x=events["timestamp"], y=events["hybrid_score"], mode="markers",
        marker=dict(size=8, color=colors, line=dict(width=1, color="white")),
        text=events["meter_id"] + " | " + events["anomaly_type"],
        hovertemplate="%{x}<br>score: %{y:.3f}<br>%{text}<extra></extra>",
        name="Flagged events",
    ))
    fig.update_layout(
        title="Flagged Anomaly Events on Held-Out Test Period (red = confirmed injected anomaly)",
        xaxis_title="Timestamp", yaxis_title="Hybrid anomaly score",
        template=TEMPLATE, height=380, showlegend=False,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def build_daily_fig(daily: pd.DataFrame):
    agg = daily.groupby("date").agg(
        avg_score=("avg_hybrid_score", "mean"),
        flagged=("flagged_readings", "sum"),
    ).reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=agg["date"], y=agg["avg_score"], name="Avg hybrid score",
                              line=dict(color=COLORS["accent"])), secondary_y=False)
    fig.add_trace(go.Bar(x=agg["date"], y=agg["flagged"], name="Flagged readings",
                          marker_color="#94a3b8", opacity=0.5), secondary_y=True)
    fig.update_layout(
        title="Fleet-Wide Daily Anomaly Score & Flagged Volume (Operational View)",
        template=TEMPLATE, height=380, margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_yaxes(title_text="Avg hybrid score", secondary_y=False)
    fig.update_yaxes(title_text="Flagged readings", secondary_y=True)
    return fig


def build_governance_table(gov: pd.DataFrame):
    gov_display = gov.copy()
    for c in ["threshold", "precision", "recall", "f1", "roc_auc"]:
        gov_display[c] = gov_display[c].apply(lambda v: f"{v:.3f}" if pd.notnull(v) else "-")
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(gov_display.columns), fill_color="#1e293b", font=dict(color="white"), align="left"),
        cells=dict(values=[gov_display[c] for c in gov_display.columns],
                   fill_color=[["#f8fafc", "#eef2f7"] * (len(gov_display) // 2 + 1)], align="left"),
    )])
    fig.update_layout(title="Model Governance Log", template=TEMPLATE, height=300,
                       margin=dict(l=10, r=10, t=50, b=10))
    return fig


def render_html(cards, figs):
    card_html = "".join(
        f'<div class="card"><div class="card-label">{label}</div>'
        f'<div class="card-value">{value}</div></div>'
        for label, value in cards
    )
    chart_divs = ""
    for i, fig in enumerate(figs):
        html_piece = pio.to_html(fig, include_plotlyjs=(i == 0), full_html=False, config={"displaylogo": False})
        chart_divs += f'<div class="chart">{html_piece}</div>'
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>VoltWatch — Smart Meter Anomaly Detection Dashboard</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin:0; background:#f1f5f9; color:#0f172a; }}
  header {{ background:{COLORS['bg']}; color:white; padding:28px 40px; }}
  header h1 {{ margin:0; font-size:26px; }}
  header p {{ margin:6px 0 0; color:#94a3b8; font-size:14px; }}
  .cards {{ display:flex; flex-wrap:wrap; gap:16px; padding:24px 40px 0; }}
  .card {{ background:white; border-radius:10px; padding:16px 20px; box-shadow:0 1px 3px rgba(0,0,0,0.08); flex:1; min-width:170px; }}
  .card-label {{ font-size:12px; color:#64748b; text-transform:uppercase; letter-spacing:0.04em; }}
  .card-value {{ font-size:24px; font-weight:700; color:{COLORS['accent']}; margin-top:4px; }}
  .content {{ padding:24px 40px 60px; max-width:1400px; margin:0 auto; }}
  .chart {{ background:white; border-radius:10px; padding:8px; margin-bottom:24px; box-shadow:0 1px 3px rgba(0,0,0,0.08); }}
  footer {{ text-align:center; color:#94a3b8; font-size:12px; padding:20px; }}
</style>
</head>
<body>
<header>
  <h1>VoltWatch — Hybrid Smart-Meter Anomaly Detection</h1>
  <p>Isolation Forest + LSTM Autoencoder · Fleet drift analysis & model governance · held-out test set results</p>
</header>
<div class="cards">{card_html}</div>
<div class="content">
  {chart_divs}
</div>
<footer>Generated from real evaluation results (outputs/eval_results.json). Not resume placeholders — see README for methodology.</footer>
</body>
</html>"""


def run():
    with open(f"{OUT_DIR}/eval_results.json") as f:
        eval_results = json.load(f)
    drift = pd.read_csv(f"{DASH_DATA}/drift_metrics.csv")
    events = pd.read_csv(f"{DASH_DATA}/anomaly_events.csv", parse_dates=["timestamp"])
    daily = pd.read_csv(f"{DASH_DATA}/daily_meter_summary.csv")
    gov = pd.read_csv(f"{DASH_DATA}/model_governance_log.csv")

    cards = kpi_cards(eval_results)
    figs = [build_drift_fig(drift), build_events_fig(events), build_daily_fig(daily), build_governance_table(gov)]

    import os
    os.makedirs(f"{OUT_DIR}/dashboard", exist_ok=True)
    html = render_html(cards, figs)
    with open(DASH_OUT, "w") as f:
        f.write(html)
    print(f"Saved dashboard -> {DASH_OUT}")


if __name__ == "__main__":
    run()
