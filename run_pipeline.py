"""
VoltWatch - Full Pipeline Orchestrator
=========================================
Runs the entire pipeline end to end:
  1. ETL ingestion (raw -> cleaned -> synthetic fleet -> resampled)
  2. Synthetic anomaly injection (labeled evaluation set)
  3. Feature engineering (rolling stats + Fourier/spectral features)
  4. Evaluation (Isolation Forest CV + LSTM autoencoder holdout + hybrid fusion)
  5. Dashboard data export (Power BI / Tableau-ready CSVs)


Usage:
    python run_pipeline.py
"""
import subprocess
import sys
import time

STEPS = [
    ("ETL: ingest + clean + build fleet + resample", "src/etl/ingest.py"),
    ("Inject synthetic labeled anomalies", "src/etl/inject_anomalies.py"),
    ("Feature engineering", "src/etl/features.py"),
    ("Model training + evaluation (CV + holdout)", "src/evaluation/evaluate.py"),
    ("Export Power BI / Tableau data marts", "src/dashboards/export_data.py"),
    
]


def main():
    for label, script in STEPS:
        print(f"\n{'='*70}\n{label}\n{'='*70}")
        t0 = time.time()
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"FAILED at step: {label}")
            sys.exit(1)
        print(f"[{time.time()-t0:.1f}s]")

    print("\nPipeline complete.")
    print("  BI extracts: outputs/dashboard_data/")
    print("  Eval results: outputs/eval_results.json")


if __name__ == "__main__":
    main()
