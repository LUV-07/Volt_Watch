# VoltWatch — Hybrid Smart-Meter Anomaly Detection

A hybrid anomaly detection system (Isolation Forest + LSTM autoencoder) for
energy theft and equipment/sensor fault detection on smart meter time-series
data, with a full ETL pipeline, honest evaluation, and BI-ready dashboards.

**Stack:** Python, PyTorch (LSTM autoencoder), scikit-learn (Isolation Forest),
pandas/pyarrow (ETL), Plotly (dashboard) — data marts exported as CSV for
Power BI / Tableau.

## Why this README is upfront about the numbers

This project also answers a question worth asking of any resume bullet:
**are the metrics backed by a real test, or are they a guess?** Every
number below comes straight from `outputs/eval_results.json`, produced by
`src/evaluation/evaluate.py`. Nothing here is hand-picked.

## Data

Source: [UCI Household Power Consumption dataset](https://archive.ics.uci.edu/dataset/235)
(CC BY 4.0) — 2,075,259 real minute-level readings from one household in
France, Dec 2006–Nov 2010. It has no theft/fault labels, so two things were
built on top, both disclosed rather than hidden:
- **Synthetic fleet**: the one real trace is expanded into 6 simulated
  meters (scale/phase/noise varied per meter) to look like a small fleet.
- **Injected anomalies**: synthetic theft-like (sustained drop, zero-
  flatline) and fault-like (voltage sag/spike, sensor spike) events are
  injected onto held-out windows with ground-truth labels — standard
  practice given how scarce real confirmed theft/fault labels are.

## What the ETL actually does

`src/etl/ingest.py` loads the raw CSV from a local folder structured like an
S3 raw zone, forward-fills short gaps (~1.25% missing), clips
physically-impossible voltage readings, builds the synthetic 6-meter fleet,
then resamples 1-min → 15-min readings (matches real meter intervals and
cut the working dataset from 12.3M rows down to 830K, which is what made
the rest of the pipeline fast). `src/etl/inject_anomalies.py` then injects
the labeled synthetic anomalies described above, and `src/etl/features.py`
builds rolling stats (1h/4h/24h mean, std) and Fourier/spectral features
(dominant frequency, spectral entropy) per meter on top of that.

## Models

- **Isolation Forest** — point-in-time detector on the engineered features.
- **LSTM Autoencoder** — trained only on windows labeled normal, so
  reconstruction error on anomalous windows is a genuine "doesn't fit the
  pattern" signal, not memorization.
- **Hybrid** — weighted average of both scores, plus a solo-high-confidence
  override so one model's blind spot can't suppress the other's catch.

## Evaluation methodology

1. Last 20% of each meter's timeline held out as a final test set, never
   touched during training.
2. Isolation Forest: 4-fold expanding-window time-series CV on the rest.
3. LSTM: trained once on the train pool (retraining a sequence model per
   fold wasn't practical here — noted, not hidden).
4. Both models retrained on the full train pool, scored on the untouched
   test set, reported at the F1-optimal threshold and a 90%-precision-floor
   threshold (the more realistic "don't cry wolf" ops setting).

## Results (real, not placeholders)

| Model | ROC-AUC | P @ best F1 | R @ best F1 | P @ 90% floor | R @ 90% floor |
|---|---|---|---|---|---|
| Isolation Forest | 0.887 | 0.286 | 0.270 | 1.000 | 0.004 |
| LSTM Autoencoder | 0.687 | 0.501 | 0.243 | 0.902 | 0.127 |
| **Hybrid** | **0.874** | **0.384** | **0.290** | **0.900** | **0.133** |

Not "91% precision / 88% recall" — those were a guess before testing.
ROC-AUC in the high 0.8s means the models rank anomalies well above normal
readings; precision/recall are modest given how rare (0.57%) and short the
injected events are relative to fleet noise. This is the version worth
defending in an interview.

## Dashboards

- `outputs/dashboard/voltwatch_dashboard.html` — open directly in a
  browser, no server needed.
- `outputs/dashboard_data/*.csv` — drop straight into Power BI
  (*Get Data → Text/CSV*) or Tableau (*Text File*), zero transformation.

## Project structure

```
src/etl/            ingest.py, inject_anomalies.py, features.py
src/models/          isolation_forest_model.py, lstm_autoencoder.py, hybrid_detector.py
src/evaluation/      evaluate.py (CV + holdout)
src/dashboards/      export_data.py, build_dashboard.py
run_pipeline.py      end-to-end orchestrator
outputs/             eval_results.json, dashboard/, dashboard_data/
```

## Reproducing

```bash
pip install -r requirements.txt
python run_pipeline.py
```
~6-8 min on CPU. Place the raw UCI file at
`data/lake_sim/raw/household_power_consumption.txt` first (not committed,
to keep the repo light).

## Honest limitations

- Fleet is derived from one real household, not independently metered homes.
- Anomalies are synthetically injected, not confirmed real cases.
- No real AWS S3 (by design, to avoid cloud costs) — ETL is structured so a
  real S3 reader is a drop-in swap.
- LSTM wasn't cross-validated, only holdout-tested.
