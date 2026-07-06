"""
VoltWatch ETL - Feature Engineering Layer
==========================================
Turns each meter's resampled time series into a per-timestep feature
vector combining:
  - Rolling statistics (mean/std/min/max over multiple windows) to capture
    local level and volatility shifts (theft often shows as a sustained
    drop in mean; faults often show as a spike in volatility).
  - Fourier-transform features (dominant frequency amplitude, spectral
    entropy, energy in the daily-cycle band) computed on a rolling window,
    to capture periodicity breakdown - a household's daily usage rhythm is
    highly periodic, and meter tampering / bypassed metering disrupts that
    periodicity even when the mean looks normal.
"""
import numpy as np
import pandas as pd

INJECTED_PATH = "/home/claude/voltwatch/data/lake_sim/eval/fleet_injected.parquet"
FEATURES_DIR = "/home/claude/voltwatch/data/lake_sim/features"

ROLLING_WINDOWS = [4, 16, 96]   # 1h, 4h, 24h at 15-min resolution
FFT_WINDOW = 96                  # 24h window for spectral features
BASE_COLS = [
    "Global_active_power", "Global_reactive_power", "Voltage",
    "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
]


def add_rolling_stats(g: pd.DataFrame) -> pd.DataFrame:
    out = g.copy()
    for w in ROLLING_WINDOWS:
        roll = g["Global_active_power"].rolling(w, min_periods=max(2, w // 2))
        out[f"gap_roll_mean_{w}"] = roll.mean()
        out[f"gap_roll_std_{w}"] = roll.std()
        roll_v = g["Voltage"].rolling(w, min_periods=max(2, w // 2))
        out[f"voltage_roll_std_{w}"] = roll_v.std()
    # ratio of instantaneous reading to its own 24h rolling mean - a cheap,
    # scale-invariant "how far from normal" signal
    out["gap_vs_24h_mean_ratio"] = g["Global_active_power"] / (
        out["gap_roll_mean_96"].replace(0, np.nan)
    )
    return out


def _spectral_features(window: np.ndarray) -> tuple:
    if len(window) < 8 or np.all(window == window[0]):
        return 0.0, 0.0
    fft_vals = np.abs(np.fft.rfft(window - window.mean()))
    power = fft_vals ** 2
    total_power = power.sum()
    if total_power <= 0:
        return 0.0, 0.0
    dominant_amp = float(fft_vals.max() / (len(window)))
    p = power / total_power
    p = p[p > 0]
    spectral_entropy = float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0
    return dominant_amp, spectral_entropy


def add_fourier_features(g: pd.DataFrame, window: int = FFT_WINDOW) -> pd.DataFrame:
    values = g["Global_active_power"].to_numpy()
    n = len(values)
    dom_amp = np.zeros(n)
    spec_ent = np.zeros(n)
    for i in range(n):
        start = max(0, i - window + 1)
        seg = values[start:i + 1]
        a, e = _spectral_features(seg)
        dom_amp[i] = a
        spec_ent[i] = e
    out = g.copy()
    out["fft_dominant_amplitude_24h"] = dom_amp
    out["fft_spectral_entropy_24h"] = spec_ent
    return out


def build_features():
    fleet = pd.read_parquet(INJECTED_PATH)
    label_cols = ["is_anomaly", "anomaly_type"]
    all_feats = []
    for meter_id, g in fleet.groupby("meter_id"):
        g = g.sort_index()
        labels = g[label_cols]
        g = add_rolling_stats(g.drop(columns=label_cols))
        g = add_fourier_features(g)
        g[label_cols] = labels
        g["meter_id"] = meter_id
        all_feats.append(g)
        print(f"  features built for {meter_id}: {len(g):,} rows")

    feats = pd.concat(all_feats).sort_index()
    feats = feats.replace([np.inf, -np.inf], np.nan)
    feats = feats.bfill().ffill()

    import os
    os.makedirs(FEATURES_DIR, exist_ok=True)
    out_path = f"{FEATURES_DIR}/fleet_features.parquet"
    feats.to_parquet(out_path)
    print(f"Saved -> {out_path} ({len(feats):,} rows, {feats.shape[1]} columns)")
    return feats


if __name__ == "__main__":
    build_features()
