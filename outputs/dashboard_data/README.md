# Dashboard Data Marts — Column Reference

Import directly into Power BI (`Get Data → Text/CSV`) or Tableau
(`Text File` connector). No transformation needed.

## anomaly_events.csv (grain: one row per flagged reading)
| Column | Description |
|---|---|
| timestamp | Reading timestamp |
| date / hour | Derived date and hour-of-day |
| meter_id | Simulated meter identifier (MTR-001 … MTR-006) |
| hybrid_score / if_score / lstm_score | Normalized [0,1] anomaly scores |
| true_positive | Whether this matches a known injected anomaly (evaluation only — a real deployment wouldn't have this column) |
| anomaly_type | Injected anomaly type, or `none` if a false positive |
| detected_type_guess | Confidence bucket derived from hybrid_score |

## daily_meter_summary.csv (grain: one row per meter-day)
Readings count, avg/max hybrid score, flagged count and rate, true anomaly count — for daily operational rollups.

## drift_metrics.csv (grain: one row per meter-month)
Avg IF/LSTM/hybrid score, score std, and `score_drift_pct` (% change vs.
that meter's own first month) — feeds the drift-analysis chart.

## model_governance_log.csv (grain: one row per model + operating point)
Threshold, precision, recall, F1, ROC-AUC for each model at each tuned
operating point (F1-optimal vs. 90%-precision floor) — for model
governance / audit reporting.
