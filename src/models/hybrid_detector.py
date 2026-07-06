"""
VoltWatch - Hybrid Detector
=============================
Combines the two detectors' normalized [0,1] anomaly scores:
  - Isolation Forest: strong on magnitude-based, point-in-time anomalies
    (voltage sags/spikes, flatlines, sudden drops).
  - LSTM autoencoder: strong on temporal/contextual anomalies (right
    magnitude, wrong time-of-day pattern; periodicity breakdown).

Fusion strategy: weighted max-mean hybrid. We take a weighted average of
both scores (tunable weights, default equal) AND flag anything either
model scores very highly on its own - a hybrid should not let one model's
blind spot suppress the other's confident detection.
"""
import numpy as np
import pandas as pd

FEATURES_PATH = "/home/claude/voltwatch/data/lake_sim/features/fleet_features.parquet"


def hybrid_score(if_scores: np.ndarray, lstm_scores: np.ndarray,
                  w_if: float = 0.5, w_lstm: float = 0.5,
                  solo_confidence_threshold: float = 0.97) -> np.ndarray:
    weighted = w_if * if_scores + w_lstm * lstm_scores
    solo_flag = np.maximum(
        np.where(if_scores > solo_confidence_threshold, if_scores, 0),
        np.where(lstm_scores > solo_confidence_threshold, lstm_scores, 0),
    )
    return np.maximum(weighted, solo_flag)


def choose_threshold(scores: np.ndarray, labels: np.ndarray, target_precision: float = None):
    """Pick an operating threshold. If target_precision is given, choose the
    lowest threshold that still meets it (maximizes recall subject to a
    precision floor) - the realistic way an ops team would tune this dial,
    rather than picking the threshold that happens to make F1 look best."""
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if target_precision is not None:
        valid = precision[:-1] >= target_precision
        if valid.any():
            best_idx = np.where(valid)[0][np.argmax(recall[:-1][valid])]
            return thresholds[best_idx], precision[best_idx], recall[best_idx]
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = np.argmax(f1[:-1])
    return thresholds[best_idx], precision[best_idx], recall[best_idx]
