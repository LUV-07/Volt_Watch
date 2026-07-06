"""
VoltWatch ETL - Ingestion Layer
================================
Simulates ingesting raw smart-meter readings from an S3-style data lake
(local filesystem stands in for `s3://voltwatch-raw/meters/...` so the
pipeline can be pointed at real S3 later by swapping `LakePath` for a
boto3-backed reader without touching downstream code).

Source data: UCI "Individual Household Electric Power Consumption" dataset
(Hebrail & Berard, 2006-2010, CC BY 4.0) - 2,075,259 minute-level readings
from a single household in Sceaux, France, Dec 2006 - Nov 2010.

To demonstrate fleet-level theft/fault detection (rather than a single
meter), the base signal is used to derive a small synthetic fleet of
meters via documented, seeded perturbations (scale, phase-shift, noise).
This is disclosed in the README - it is NOT presented as independently
metered households.
"""
import os
import numpy as np
import pandas as pd

RAW_PATH = "/home/claude/voltwatch/data/lake_sim/raw/household_power_consumption.txt"
PROCESSED_DIR = "/home/claude/voltwatch/data/lake_sim/processed"

N_FLEET_METERS = 6          # size of the simulated meter fleet
RESAMPLE_FREQ = "15min"      # working granularity for feature engineering/modeling
RANDOM_SEED = 42


def load_raw() -> pd.DataFrame:
    """Ingest the raw minute-level CSV exactly as it would land in the raw
    zone of a data lake (semicolon-delimited, '?' = missing)."""
    df = pd.read_csv(
        RAW_PATH,
        sep=";",
        na_values=["?"],
        low_memory=False,
    )
    df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)
    df = df.drop(columns=["Date", "Time"]).set_index("timestamp").sort_index()
    numeric_cols = [
        "Global_active_power", "Global_reactive_power", "Voltage",
        "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[numeric_cols]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Handle the ~1.25% missing readings via short forward-fill (meter
    outages of a few minutes) capped at 5 minutes; anything longer is left
    NaN and dropped so we never invent long stretches of fake readings."""
    df = df.copy()
    df = df.ffill(limit=5)
    df = df.dropna()
    # physically implausible readings (sensor faults) get clipped, not dropped,
    # so genuine fault signatures survive into the anomaly detector
    df["Voltage"] = df["Voltage"].clip(lower=150, upper=280)
    df["Global_active_power"] = df["Global_active_power"].clip(lower=0)
    return df


def build_fleet(df: pd.DataFrame, n_meters: int = N_FLEET_METERS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Derive a synthetic multi-meter fleet from the single real household
    trace via seeded, documented perturbations. Each meter keeps the real
    signal's shape (daily/seasonal structure) but differs in baseline load,
    phase, and measurement noise - representative of how different
    households on the same feeder differ."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_meters):
        meter_id = f"MTR-{i+1:03d}"
        scale = rng.uniform(0.6, 1.8)
        phase_shift = int(rng.integers(0, 1440))  # minutes
        noise_std = rng.uniform(0.01, 0.04)

        shifted = df.shift(phase_shift).bfill()
        f = shifted.copy()
        f["Global_active_power"] *= scale
        f["Global_intensity"] *= scale
        f["Sub_metering_1"] *= rng.uniform(0.7, 1.3)
        f["Sub_metering_2"] *= rng.uniform(0.7, 1.3)
        f["Sub_metering_3"] *= rng.uniform(0.7, 1.3)
        f["Voltage"] += rng.normal(0, 1.5, size=len(f))

        noise = rng.normal(0, noise_std, size=len(f)) * f["Global_active_power"].std()
        f["Global_active_power"] = (f["Global_active_power"] + noise).clip(lower=0)

        f["meter_id"] = meter_id
        frames.append(f)

    fleet = pd.concat(frames).sort_index()
    return fleet


def resample_fleet(fleet: pd.DataFrame, freq: str = RESAMPLE_FREQ) -> pd.DataFrame:
    """Resample each meter's minute-level trace to a working granularity.
    Real deployments read smart meters every 15/30/60 min, not every
    minute, so this both matches real-world practice and keeps downstream
    modeling tractable."""
    out = []
    for meter_id, g in fleet.groupby("meter_id"):
        r = g.drop(columns="meter_id").resample(freq).mean()
        r["meter_id"] = meter_id
        out.append(r)
    return pd.concat(out).sort_index()


def run():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print("Ingesting raw minute-level readings from data lake (raw zone)...")
    raw = load_raw()
    print(f"  {len(raw):,} raw minute-level rows ingested")

    print("Cleaning (missing-value handling, physical bounds)...")
    cleaned = clean(raw)
    print(f"  {len(cleaned):,} rows after cleaning ({100*(1-len(cleaned)/len(raw)):.2f}% dropped)")

    print(f"Deriving synthetic {N_FLEET_METERS}-meter fleet from base signal...")
    fleet = build_fleet(cleaned)
    print(f"  {len(fleet):,} total minute-level fleet rows")

    print(f"Resampling fleet to {RESAMPLE_FREQ} working granularity...")
    resampled = resample_fleet(fleet, RESAMPLE_FREQ)
    print(f"  {len(resampled):,} rows at {RESAMPLE_FREQ} resolution")

    out_path = os.path.join(PROCESSED_DIR, "fleet_resampled.parquet")
    resampled.to_parquet(out_path)
    print(f"Saved -> {out_path}")

    manifest = {
        "raw_rows_ingested": len(raw),
        "rows_after_cleaning": len(cleaned),
        "fleet_meters": N_FLEET_METERS,
        "fleet_minute_rows": len(fleet),
        "resample_freq": RESAMPLE_FREQ,
        "resampled_rows": len(resampled),
    }
    return manifest


if __name__ == "__main__":
    print(run())
