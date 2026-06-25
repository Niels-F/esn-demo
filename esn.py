"""Minimal Echo State Network (ESN) — one-step-ahead prediction demo.

A reservoir of random recurrent neurons is driven by an input signal; only a
linear readout is trained (ridge regression). Classic, tiny, dependency-light.
"""

import os
import numpy as np

rng = np.random.default_rng(42)

# --- Target signal: a smooth, quasi-periodic series ---------------------------
T = 3000
t = np.arange(T)
signal = np.sin(0.10 * t) + 0.5 * np.sin(0.21 * t) + 0.3 * np.sin(0.43 * t)
signal = signal / np.max(np.abs(signal))
data = signal.reshape(-1, 1)

# --- Hyper-parameters ---------------------------------------------------------
n_in, n_res = 1, 300
spectral_radius = 0.95
sparsity = 0.10
reg = 1e-6
washout = 100
train_len, test_len = 2000, 800

# --- Reservoir weights --------------------------------------------------------
Win = rng.uniform(-1, 1, (n_res, n_in + 1))           # +1 for bias
W = rng.uniform(-1, 1, (n_res, n_res))
W *= rng.uniform(0, 1, (n_res, n_res)) < sparsity     # make it sparse
W *= spectral_radius / np.max(np.abs(np.linalg.eigvals(W)))

def step(x, u):
    u_aug = np.vstack(([[1.0]], u.reshape(-1, 1)))    # bias + input
    return np.tanh(Win @ u_aug + W @ x)

# --- Collect reservoir states over the training window ------------------------
x = np.zeros((n_res, 1))
X, Y = [], []
for i in range(train_len):
    x = step(x, data[i])
    if i >= washout:
        X.append(np.vstack(([[1.0]], data[i].reshape(-1, 1), x)))
        Y.append(data[i + 1].reshape(-1, 1))
X, Y = np.hstack(X), np.hstack(Y)

# --- Train the linear readout (ridge regression) ------------------------------
Wout = Y @ X.T @ np.linalg.inv(X @ X.T + reg * np.eye(X.shape[0]))

# --- Test: one-step-ahead prediction -----------------------------------------
preds, truth = [], []
for i in range(train_len, train_len + test_len):
    x = step(x, data[i])
    feat = np.vstack(([[1.0]], data[i].reshape(-1, 1), x))
    preds.append(float((Wout @ feat).item()))
    truth.append(float(data[i + 1].item()))
preds, truth = np.array(preds), np.array(truth)

mse = np.mean((preds - truth) ** 2)
print(f"Reservoir size: {n_res}   Test MSE: {mse:.3e}")

# --- Plot ---------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
plt.figure(figsize=(10, 4))
plt.plot(truth[:200], label="target", linewidth=2)
plt.plot(preds[:200], "--", label="ESN prediction", linewidth=2)
plt.title(f"Echo State Network — one-step-ahead (test MSE = {mse:.1e})")
plt.xlabel("time step")
plt.ylabel("signal")
plt.legend()
plt.tight_layout()
plt.savefig("figures/prediction.png", dpi=110)
print("saved figures/prediction.png")
