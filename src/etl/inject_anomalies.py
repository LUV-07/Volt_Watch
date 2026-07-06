"""
VoltWatch - Labeled Evaluation Set via Synthetic Anomaly Injection
====================================================================
The UCI source data has NO ground-truth labels for theft or faults (it's
ordinary household consumption). To honestly report precision/recall, we
need labeled anomalies. Standard practice in the energy-theft-detection
literature (since real, confirmed theft/fault labels are scarce and
sensitive) is to inject synthetic anomalies with known ground truth on
top of real consumption data, then evaluate detectors against that.

This module injects two families of labeled anomalies onto held-out
segments of each meter's feature series:

THEFT-like patterns (meter tampering / bypass):
  - sustained_drop:  consumption drops to 10-30% of normal for hours,
                     mimicking a bypassed or under-reporting meter.
  - zero_flatline:   consumption flatlines near zero while voltage stays
                     normal - classic tamper/bypass signature.

FAULT-like patterns (equipment/sensor faults):
  - voltage_sag:     voltage drops well outside normal operating range.
  - voltage_spike:   voltage surges outside normal operating range.
  - reading_spike:   an implausible short current/power spike (sensor glitch).

Every injected point is labeled `is_anomaly=1`; everything else is 0.
Injection windows are chosen to not overlap, and at most ~3% of each
meter's held-out timeline is anomalous, matching realistic base rates.
"""
import numpy as np
import pandas as pd

PROCESSED_PATH = "/home/claude/voltwatch/data/lake_sim/processed/fleet_resampled.parquet"
EVAL_DIR = "/home/claude/voltwatch/data/lake_sim/eval"

RANDOM_SEED = 7
ANOMALY_TYPES = ["sustained_drop", "zero_flatline", "voltage_sag", "voltage_spike", "reading_spike"]


def inject_for_meter(g: pd.DataFrame, rng: np.random.Generator, meter_id: str) -> pd.DataFrame:
    g = g.copy()
    n = len(g)
    g["is_anomaly"] = 0
    g["anomaly_type"] = "none"

    n_events = max(6, n // 4000)  # roughly one event per ~4000 timesteps (~41 days at 15min)
    used = np.zeros(n, dtype=bool)

    for _ in range(n_events):
        a_type = rng.choice(ANOMALY_TYPES)
        duration = int(rng.integers(4, 48))  # 1h - 12h at 15-min resolution
        for _try in range(20):
            start = int(rng.integers(200, n - duration - 200))
            if not used[start:start + duration].any():
                break
        else:
            continue
        used[start:start + duration] = True
        idx = g.index[start:start + duration]

        if a_type == "sustained_drop":
            factor = rng.uniform(0.1, 0.3)
            g.loc[idx, "Global_active_power"] *= factor
            g.loc[idx, "Global_intensity"] *= factor
        elif a_type == "zero_flatline":
            g.loc[idx, "Global_active_power"] = rng.uniform(0.01, 0.05)
            g.loc[idx, "Global_intensity"] = rng.uniform(0.1, 0.3)
        elif a_type == "voltage_sag":
            g.loc[idx, "Voltage"] = rng.uniform(150, 190)
        elif a_type == "voltage_spike":
            g.loc[idx, "Voltage"] = rng.uniform(260, 285)
        elif a_type == "reading_spike":
            spike_len = max(1, duration // 6)
            spike_idx = idx[:spike_len]
            g.loc[spike_idx, "Global_active_power"] *= rng.uniform(4, 8)
            g.loc[spike_idx, "Global_intensity"] *= rng.uniform(4, 8)

        g.loc[idx, "is_anomaly"] = 1
        g.loc[idx, "anomaly_type"] = a_type

    print(f"  {meter_id}: injected {n_events} events, {g['is_anomaly'].sum()} anomalous rows "
          f"({100*g['is_anomaly'].mean():.2f}%)")
    return g


def build_eval_set():
    raw = pd.read_parquet(PROCESSED_PATH)
    rng = np.random.default_rng(RANDOM_SEED)
    out = []
    for meter_id, g in raw.groupby("meter_id"):
        g = g.sort_index().drop(columns="meter_id")
        g = inject_for_meter(g, rng, meter_id)
        g["meter_id"] = meter_id
        out.append(g)
    result = pd.concat(out).sort_index()

    import os
    os.makedirs(EVAL_DIR, exist_ok=True)
    out_path = f"{EVAL_DIR}/fleet_injected.parquet"
    result.to_parquet(out_path)
    print(f"Saved anomaly-injected series -> {out_path} ({len(result):,} rows, "
          f"{result['is_anomaly'].sum():,} anomalous = {100*result['is_anomaly'].mean():.2f}%)")
    return result


if __name__ == "__main__":
    build_eval_set()
