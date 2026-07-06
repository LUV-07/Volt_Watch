"""
VoltWatch - Isolation Forest Detector
=======================================
Point-in-time statistical anomaly detector. Operates on the engineered
feature vector (raw readings + rolling stats + spectral features) for
each timestep independently. Good at catching magnitude-based anomalies
(voltage sags/spikes, flatlines, sustained drops) that show up as
outliers in feature space, even without needing sequence context.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

FEATURES_PATH = "/home/claude/voltwatch/data/lake_sim/features/fleet_features.parquet"
MODEL_DIR = "/home/claude/voltwatch/models"

FEATURE_COLS = [
    "Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    "gap_roll_mean_4", "gap_roll_std_4", "voltage_roll_std_4",
    "gap_roll_mean_16", "gap_roll_std_16", "voltage_roll_std_16",
    "gap_roll_mean_96", "gap_roll_std_96", "voltage_roll_std_96",
    "gap_vs_24h_mean_ratio", "fft_dominant_amplitude_24h", "fft_spectral_entropy_24h",
]


def load_features():
    df = pd.read_parquet(FEATURES_PATH)
    return df


def train_isolation_forest(train_df: pd.DataFrame, contamination: float = 0.01, seed: int = 42,
                            n_estimators: int = 300, max_samples=0.5):
    scaler = StandardScaler()
    X = scaler.fit_transform(train_df[FEATURE_COLS])
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X)
    return model, scaler


def score(model, scaler, df: pd.DataFrame) -> np.ndarray:
    X = scaler.transform(df[FEATURE_COLS])
    # decision_function: higher = more normal. Flip sign so higher = more anomalous,
    # then min-max normalize to [0, 1] for easy fusion with the LSTM score later.
    raw = -model.decision_function(X)
    norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    return norm


def save(model, scaler, path=MODEL_DIR):
    os.makedirs(path, exist_ok=True)
    joblib.dump(model, f"{path}/isolation_forest.joblib")
    joblib.dump(scaler, f"{path}/if_scaler.joblib")


def load(path=MODEL_DIR):
    model = joblib.load(f"{path}/isolation_forest.joblib")
    scaler = joblib.load(f"{path}/if_scaler.joblib")
    return model, scaler


if __name__ == "__main__":
    df = load_features()
    model, scaler = train_isolation_forest(df)
    save(model, scaler)
    s = score(model, scaler, df)
    print(f"Trained Isolation Forest on {len(df):,} rows, {len(FEATURE_COLS)} features")
    print(f"Score range: [{s.min():.3f}, {s.max():.3f}], mean={s.mean():.3f}")
