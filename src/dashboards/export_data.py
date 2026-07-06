"""
VoltWatch - Dashboard Data Export Layer
==========================================
Produces clean, denormalized CSV extracts designed to be dropped straight
into Power BI (Get Data > Text/CSV) or Tableau (Text File connector) with
no further transformation - column names, types, and grain are chosen for
BI tools rather than for Python.

Outputs (all in outputs/dashboard_data/):
  - anomaly_events.csv     : one row per flagged anomaly (grain: event)
  - daily_meter_summary.csv: one row per meter per day (grain: meter-day)
  - drift_metrics.csv      : one row per meter per month (grain: meter-month)
  - model_governance_log.csv: one row per model version/run (grain: run)
"""
import json
import os
import numpy as np
import pandas as pd

EVAL_DIR = "/home/claude/voltwatch/outputs"
DASH_DATA_DIR = "/home/claude/voltwatch/outputs/dashboard_data"
OPERATING_THRESHOLD_KEY = "precision_90_floor"  # which tuned threshold to use for "flagged" events


def build_anomaly_events(scored: pd.DataFrame, threshold: float) -> pd.DataFrame:
    flagged = scored[scored["hybrid_score"] >= threshold].copy()
    flagged = flagged.reset_index(drop=False).rename(columns={"index": "timestamp"})
    flagged["date"] = flagged["timestamp"].dt.date
    flagged["hour"] = flagged["timestamp"].dt.hour
    flagged["true_positive"] = flagged["is_anomaly"] == 1
    flagged["detected_type_guess"] = np.where(
        flagged["hybrid_score"] > 0.85, "high_confidence", "moderate_confidence"
    )
    cols = ["timestamp", "date", "hour", "meter_id", "hybrid_score", "if_score",
            "lstm_score", "true_positive", "anomaly_type", "detected_type_guess"]
    return flagged[cols].sort_values("timestamp")


def build_daily_summary(scored: pd.DataFrame, threshold: float) -> pd.DataFrame:
    scored = scored.copy()
    scored["date"] = scored.index.date
    scored["flagged"] = (scored["hybrid_score"] >= threshold).astype(int)
    daily = scored.groupby(["meter_id", "date"]).agg(
        readings=("hybrid_score", "count"),
        avg_hybrid_score=("hybrid_score", "mean"),
        max_hybrid_score=("hybrid_score", "max"),
        flagged_readings=("flagged", "sum"),
        true_anomalies=("is_anomaly", "sum"),
    ).reset_index()
    daily["flagged_rate_pct"] = 100 * daily["flagged_readings"] / daily["readings"]
    return daily


def build_drift_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    scored = scored.copy()
    scored["month"] = scored.index.to_period("M").astype(str)
    drift = scored.groupby(["meter_id", "month"]).agg(
        avg_if_score=("if_score", "mean"),
        avg_lstm_score=("lstm_score", "mean"),
        avg_hybrid_score=("hybrid_score", "mean"),
        score_std=("hybrid_score", "std"),
        n_readings=("hybrid_score", "count"),
    ).reset_index()
    # drift signal: change in avg hybrid score vs. each meter's own first month
    drift["baseline_score"] = drift.groupby("meter_id")["avg_hybrid_score"].transform("first")
    drift["score_drift_pct"] = 100 * (drift["avg_hybrid_score"] - drift["baseline_score"]) / (
        drift["baseline_score"] + 1e-9
    )
    return drift


def build_model_governance_log(eval_results: dict) -> pd.DataFrame:
    rows = []
    for model_name, r in eval_results["holdout_test"].items():
        for regime in ["f1_optimal", "precision_90_floor"]:
            rows.append({
                "model": model_name,
                "operating_point": regime,
                "threshold": r[regime]["threshold"],
                "precision": r[regime]["precision"],
                "recall": r[regime]["recall"],
                "f1": r[regime]["f1"],
                "roc_auc": r["roc_auc"],
                "test_set_size": eval_results["test_set_size"],
                "test_set_anomaly_rate": eval_results["test_set_anomaly_rate"],
            })
    cv = eval_results["cv_isolation_forest"]
    rows.append({
        "model": "isolation_forest_cv", "operating_point": "4fold_cv_mean",
        "threshold": None,
        "precision": cv["mean"]["precision"], "recall": cv["mean"]["recall"],
        "f1": cv["mean"]["f1"], "roc_auc": cv["mean"]["roc_auc"],
        "test_set_size": None, "test_set_anomaly_rate": None,
    })
    return pd.DataFrame(rows)


def run():
    with open(f"{EVAL_DIR}/eval_results.json") as f:
        eval_results = json.load(f)
    scored = pd.read_parquet(f"{EVAL_DIR}/scored_test_set.parquet")

    threshold = eval_results["holdout_test"]["hybrid"][OPERATING_THRESHOLD_KEY]["threshold"]

    os.makedirs(DASH_DATA_DIR, exist_ok=True)
    events = build_anomaly_events(scored, threshold)
    events.to_csv(f"{DASH_DATA_DIR}/anomaly_events.csv", index=False)
    print(f"anomaly_events.csv: {len(events):,} rows")

    daily = build_daily_summary(scored, threshold)
    daily.to_csv(f"{DASH_DATA_DIR}/daily_meter_summary.csv", index=False)
    print(f"daily_meter_summary.csv: {len(daily):,} rows")

    drift = build_drift_metrics(scored)
    drift.to_csv(f"{DASH_DATA_DIR}/drift_metrics.csv", index=False)
    print(f"drift_metrics.csv: {len(drift):,} rows")

    gov = build_model_governance_log(eval_results)
    gov.to_csv(f"{DASH_DATA_DIR}/model_governance_log.csv", index=False)
    print(f"model_governance_log.csv: {len(gov):,} rows")

    print(f"\nAll BI extracts saved to {DASH_DATA_DIR}/")
    print("Import via Power BI: Get Data > Text/CSV. Tableau: Text File connector.")


if __name__ == "__main__":
    run()
