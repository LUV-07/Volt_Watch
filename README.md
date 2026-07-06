# VoltWatch — Hybrid Smart-Meter Anomaly Detection

A hybrid anomaly detection system (Isolation Forest + LSTM autoencoder) for
energy theft and equipment/sensor fault detection on smart meter time-series
data, with a full ETL pipeline, rigorous evaluation methodology, and
BI-ready dashboard exports.

**Stack:** Python, PyTorch (LSTM autoencoder), scikit-learn (Isolation Forest),
pandas/pyarrow (ETL), Plotly (interactive dashboard) — data marts exported as
CSV for Power BI / Tableau.

---

## Why this README is longer than usual

This project exists partly to answer a specific question honestly: **are the
metrics on my resume backed by a real test, or are they impressionistic?**
Every number below comes from `outputs/eval_results.json`, produced by
`src/evaluation/evaluate.py`, which anyone can re-run with
`python run_pipeline.py`. Nothing here is hand-picked.

## Data

Source: [UCI "Individual Household Electric Power Consumption"](https://archive.ics.uci.edu/dataset/235)
(Hebrail & Berard, CC BY 4.0) — **2,075,259 real minute-level readings** from
a household in Sceaux, France, Dec 2006–Nov 2010. This dataset has no theft
or fault labels (it's ordinary consumption), so two things had to be built
on top of it honestly rather than assumed away:

1. **Synthetic fleet expansion** (`src/etl/ingest.py`): the single real
   trace is expanded into a 6-meter fleet via seeded, documented
   perturbations (baseline scale, phase shift, measurement noise) — this is
   disclosed here, not presented as independently metered households.
2. **Labeled anomaly injection** (`src/etl/inject_anomalies.py`): synthetic
   theft-like patterns (sustained drop, zero-flatline/bypass) and
   fault-like patterns (voltage sag/spike, sensor reading spikes) are
   injected onto held-out time windows with ground-truth labels, following
   standard practice in the energy-theft-detection literature (confirmed
   real theft/fault labels are scarce and sensitive, so labeled injection is
   the standard substitute for a rigorous test).

## Pipeline

```
data lake (raw)          ETL                    features              models                 dashboards
─────────────────    ─────────────           ──────────────        ──────────────         ──────────────────
household_power   →  clean, build fleet,  →  rolling stats     →   Isolation Forest   →   BI data marts (CSV)
_consumption.txt     resample to 15-min       (1h/4h/24h) +         + LSTM autoencoder      + interactive HTML
(simulates S3        + inject synthetic        Fourier/spectral     + hybrid fusion         dashboard
raw zone)            labeled anomalies         features
```

Run everything: `python run_pipeline.py` (≈6–8 min on CPU; no GPU required).

- **ETL** (`src/etl/ingest.py`): loads from a local directory structured like
  an S3 raw zone (swap in `boto3` later without touching downstream code),
  handles the dataset's ~1.25% missing readings via short forward-fill,
  clips physically-implausible voltage readings, builds the synthetic fleet,
  resamples 1-min → 15-min (matches real smart-meter read intervals and
  keeps modeling tractable — this cut a full run's preprocessing time from
  ~4 min to ~90 sec by working on 830K rows instead of 12.3M).
- **Feature engineering** (`src/etl/features.py`): rolling mean/std/vs-24h-mean
  ratio at three windows, plus Fourier-transform features (dominant
  amplitude, spectral entropy) over a rolling 24h window — theft/tampering
  often disrupts a household's daily periodicity even when raw magnitude
  looks unremarkable.
- **Models**:
  - `src/models/isolation_forest_model.py` — point-in-time detector on the
    engineered feature vector.
  - `src/models/lstm_autoencoder.py` — sequence detector; trained only on
    windows labeled normal, so reconstruction error on anomalous windows is
    a genuine "this doesn't fit the learned pattern" signal, not overfit
    memorization.
  - `src/models/hybrid_detector.py` — weighted-average fusion of both
    normalized scores, plus a "solo high-confidence" override so one
    model's blind spot can't suppress the other's confident catch.

## Evaluation methodology

1. **Chronological split per meter**: last 20% of each meter's timeline held
   out as a **final test set never touched during training or tuning**.
2. **Isolation Forest**: 4-fold expanding-window time-series
   cross-validation on the remaining 80% (`TimeSeriesSplit`), reported as
   mean ± std across folds.
3. **LSTM autoencoder**: trained once on the train pool (retraining a
   sequence model 4× per fold wasn't compute-tractable in this environment
   — noted here rather than silently skipped).
4. **Final holdout evaluation**: both models retrained on the full train
   pool, scored on the untouched test set, reported at two operating
   points — the F1-optimal threshold, and a threshold tuned to a 90%
   precision floor (the more realistic ops setting: an alert queue that's
   usually wrong gets ignored).

## Results (real, from `outputs/eval_results.json`)

**Isolation Forest, 4-fold time-series CV** (24 meter-folds):

| Metric | Mean | Std |
|---|---|---|
| Precision | 0.268 | 0.167 |
| Recall | 0.279 | 0.127 |
| F1 | 0.234 | 0.110 |
| ROC-AUC | 0.839 | 0.106 |

**Final holdout test** (166,026 readings, 0.57% true anomaly rate):

| Model | ROC-AUC | P @ F1-optimal | R @ F1-optimal | P @ 90% floor | R @ 90% floor |
|---|---|---|---|---|---|
| Isolation Forest | 0.887 | 0.286 | 0.270 | 1.000 | 0.004 |
| LSTM Autoencoder | 0.687 | 0.501 | 0.243 | 0.902 | 0.127 |
| **Hybrid** | **0.874** | **0.384** | **0.290** | **0.900** | **0.133** |

**Read honestly:** ROC-AUC in the high 0.8s means the models rank anomalous
readings well above normal ones. Precision/recall at fixed thresholds are
modest — expected given how imbalanced (0.57%) and short-duration the
injected events are relative to normal fleet noise. This is a materially
different, more defensible story than a headline "91% precision / 88%
recall" — and it's the number I'd actually defend in an interview.

> **Resume note:** the original bullet's 90/88 figures were aspirational,
> not measured. If reusing this project on a resume, report the numbers
> above (or the F1-optimal precision/recall, ~38%/29% hybrid, or the
> ROC-AUC of ~0.87), not the original placeholders.

## Dashboards

- `outputs/dashboard/voltwatch_dashboard.html` — self-contained interactive
  dashboard (Plotly): drift analysis by meter/month, flagged-event
  timeline, fleet-wide daily operational view, model governance log. Open
  directly in any browser, no server needed.
- `outputs/dashboard_data/*.csv` — denormalized data marts built for
  zero-transformation import into Power BI (*Get Data → Text/CSV*) or
  Tableau (*Text File* connector):
  - `anomaly_events.csv` — one row per flagged event
  - `daily_meter_summary.csv` — one row per meter-day
  - `drift_metrics.csv` — one row per meter-month (score drift vs. each
    meter's own baseline month)
  - `model_governance_log.csv` — one row per model/operating-point,
    threshold + precision/recall/F1/ROC-AUC, for model governance
    reporting.

## Project structure

```
voltwatch/
├── data/lake_sim/          # simulated S3 raw zone → processed → features → eval
├── src/
│   ├── etl/                # ingest.py, inject_anomalies.py, features.py
│   ├── models/              # isolation_forest_model.py, lstm_autoencoder.py, hybrid_detector.py
│   ├── evaluation/          # evaluate.py (CV + holdout)
│   └── dashboards/          # export_data.py, build_dashboard.py
├── outputs/                 # eval_results.json, dashboard/, dashboard_data/
├── run_pipeline.py           # end-to-end orchestrator
└── requirements.txt
```

## Reproducing

```bash
pip install -r requirements.txt
python run_pipeline.py
```

Regenerates everything from the raw UCI file through the final dashboard.
Raw data file (`household_power_consumption.txt`, CC BY 4.0) is not
committed to keep the repo light — see the Data section above for the
source; place it at `data/lake_sim/raw/household_power_consumption.txt`
before running.

## Honest limitations

- Multi-meter "fleet" is derived from one real household, not independently
  metered homes — documented, not hidden.
- Anomalies are synthetically injected, not confirmed real theft/fault
  cases — standard practice given how scarce/sensitive real labels are, but
  still synthetic.
- LSTM autoencoder wasn't cross-validated (compute cost) — single
  chronological holdout only.
- No real AWS S3 integration in this build (by design, to avoid cloud
  costs) — the ETL layer is structured so a real S3 reader is a drop-in
  swap.
