"""
VoltWatch - LSTM Autoencoder Detector
=======================================
Sequence-based anomaly detector. Trains an encoder-decoder LSTM to
reconstruct short windows (24h at 15-min resolution = 96 steps) of
*normal* multivariate meter behavior. At inference time, windows that the
model reconstructs poorly (high MSE) are flagged as anomalous - this
catches anomalies that are subtle in any single timestep but break the
learned temporal pattern (e.g. a flatline that's individually "not that
different" from a low-usage night, but wrong for its time-of-day context).

Trained only on data the injection step marked as normal (is_anomaly==0),
so the model never learns to "expect" the injected anomalies - this is
what makes the reconstruction-error signal meaningful.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

FEATURES_PATH = "/home/claude/voltwatch/data/lake_sim/features/fleet_features.parquet"
MODEL_DIR = "/home/claude/voltwatch/models"

SEQ_LEN = 96  # 24h at 15-min resolution
SEQ_COLS = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity"]
STRIDE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WindowDataset(Dataset):
    def __init__(self, arr: np.ndarray, seq_len: int, stride: int):
        self.arr = arr
        self.seq_len = seq_len
        self.starts = list(range(0, len(arr) - seq_len + 1, stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        return torch.tensor(self.arr[s:s + self.seq_len], dtype=torch.float32)


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 32, latent_size: int = 16):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.from_latent = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, n_features)
        self.seq_len = None

    def forward(self, x):
        seq_len = x.size(1)
        _, (h, _) = self.encoder(x)
        latent = self.to_latent(h[-1])
        h0 = self.from_latent(latent).unsqueeze(1).repeat(1, seq_len, 1)
        dec_out, _ = self.decoder(h0)
        return self.output_layer(dec_out)


def fit_scaler(train_arr: np.ndarray):
    mean = train_arr.mean(axis=0)
    std = train_arr.std(axis=0) + 1e-6
    return mean, std


def train_lstm_autoencoder(df: pd.DataFrame, epochs: int = 6, batch_size: int = 256, lr: float = 1e-3):
    normal = df[df["is_anomaly"] == 0]
    arrs = []
    for meter_id, g in normal.groupby("meter_id"):
        arrs.append(g.sort_index()[SEQ_COLS].to_numpy())

    mean, std = fit_scaler(np.concatenate(arrs))

    model = LSTMAutoencoder(n_features=len(SEQ_COLS)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    all_datasets = [WindowDataset((a - mean) / std, SEQ_LEN, STRIDE) for a in arrs if len(a) > SEQ_LEN]
    combined = torch.utils.data.ConcatDataset(all_datasets)
    loader = DataLoader(combined, batch_size=batch_size, shuffle=True)

    print(f"Training LSTM autoencoder on {len(combined):,} normal windows "
          f"(seq_len={SEQ_LEN}, features={len(SEQ_COLS)}, device={DEVICE})")

    model.train()
    for epoch in range(epochs):
        total_loss, n_batches = 0.0, 0
        for batch in loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"  epoch {epoch+1}/{epochs} - reconstruction MSE: {total_loss/n_batches:.5f}")

    return model, mean, std


def reconstruction_error(model, mean, std, arr: np.ndarray, seq_len: int = SEQ_LEN,
                          stride: int = 4, batch_size: int = 512) -> np.ndarray:
    """Per-timestep anomaly score via sliding-window reconstruction error,
    averaged across all windows covering each timestep. Batched for speed."""
    model.eval()
    scaled = (arr - mean) / std
    n = len(scaled)
    err_sum = np.zeros(n)
    err_count = np.zeros(n)

    starts = list(range(0, max(1, n - seq_len + 1), stride))
    if not starts:
        return np.zeros(n)

    with torch.no_grad():
        for b_start in range(0, len(starts), batch_size):
            batch_starts = starts[b_start:b_start + batch_size]
            windows = np.stack([scaled[s:s + seq_len] for s in batch_starts])
            x = torch.tensor(windows, dtype=torch.float32).to(DEVICE)
            recon = model(x).cpu().numpy()
            per_step_err = ((recon - windows) ** 2).mean(axis=2)  # (batch, seq_len)
            for i, s in enumerate(batch_starts):
                err_sum[s:s + seq_len] += per_step_err[i]
                err_count[s:s + seq_len] += 1

    err_count[err_count == 0] = 1
    avg_err = err_sum / err_count
    norm = (avg_err - avg_err.min()) / (avg_err.max() - avg_err.min() + 1e-9)
    return norm


def save(model, mean, std, path=MODEL_DIR):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), f"{path}/lstm_autoencoder.pt")
    np.savez(f"{path}/lstm_scaler.npz", mean=mean, std=std)


def load(path=MODEL_DIR):
    model = LSTMAutoencoder(n_features=len(SEQ_COLS)).to(DEVICE)
    model.load_state_dict(torch.load(f"{path}/lstm_autoencoder.pt", map_location=DEVICE))
    scaler = np.load(f"{path}/lstm_scaler.npz")
    return model, scaler["mean"], scaler["std"]


if __name__ == "__main__":
    df = pd.read_parquet(FEATURES_PATH)
    model, mean, std = train_lstm_autoencoder(df)
    save(model, mean, std)
    print("Saved LSTM autoencoder ->", MODEL_DIR)
