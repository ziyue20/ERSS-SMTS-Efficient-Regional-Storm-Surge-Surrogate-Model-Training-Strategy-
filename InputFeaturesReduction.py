"""
Minimal input features reduction (Adaptive sampling algorithm) illustrative code
- Correlation matrix analysis
- Full vs reduced feature training
- Performance comparison

This file is intended for public GitHub release.
No raw data is included.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# User parameters
# -----------------------------
N_EPOCHS = 5
LR = 1e-3
TEST_RATIO = 0.2
RANDOM_STATE = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Placeholder data paths
# -----------------------------
TRAINING_TABLE_CSV = "path/to/FullTrainingTable.csv"

# -----------------------------
# Load data
# -----------------------------
# Assumes one row per (storm × grid point)
df = pd.read_csv(TRAINING_TABLE_CSV)

# -----------------------------
# Define feature sets
# -----------------------------
FULL_FEATURES = [
    "Heading", "V_f (knots)", "R_max (nm)", "Landfall Lon (x)", "c_p (mbar)",
    "lat", "lon", "canopy", "manning", "z0", "ele", "msl"
]

REDUCED_FEATURES = [
    "Heading", "V_f (knots)", "R_max (nm)", "Landfall Lon (x)", "c_p (mbar)",
    "lat", "lon", "manning", "ele", "msl"   # canopy and z0 removed
]

TARGET = "surge"

# -----------------------------
# 1. Correlation matrix analysis
# -----------------------------
corr_matrix = df[FULL_FEATURES].corr()
print("Correlation matrix (full features):")
print(corr_matrix.round(2))

# -----------------------------
# Helper: train + evaluate model
# -----------------------------
def train_and_evaluate(X, y):
    # Train / test split
    n_train = int((1 - TEST_RATIO) * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)

    # Training loop
    for _ in range(N_EPOCHS):
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_pred = model(X_test_t).cpu().numpy()

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r = np.corrcoef(y_test.flatten(), y_pred.flatten())[0, 1]

    return rmse, r

# -----------------------------
# 2. Train on full features
# -----------------------------
X_full = df[FULL_FEATURES].values
y = df[[TARGET]].values

rmse_full, r_full = train_and_evaluate(X_full, y)

# -----------------------------
# 3. Train on reduced features
# -----------------------------
X_reduced = df[REDUCED_FEATURES].values

rmse_reduced, r_reduced = train_and_evaluate(X_reduced, y)

# -----------------------------
# 4. Performance comparison
# -----------------------------
print("\nModel performance comparison:")
print(f"Full features   -> RMSE: {rmse_full:.4f}, R: {r_full:.4f}")
print(f"Reduced features-> RMSE: {rmse_reduced:.4f}, R: {r_reduced:.4f}")
