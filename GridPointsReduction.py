"""
Minimal grid point reduction illustrative code:
- K-means centroid extraction
- Train surrogate model
- Test performance
- Loop over number of centroids (NC)

This file is intended for public GitHub release.
No raw data is included.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# User-defined parameters
# -----------------------------
NCs = [500, 1000, 2000]          # number of centroids to test
N_EPOCHS = 5
BATCH_SIZE = 1024
LR = 1e-3
RANDOM_STATE = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Placeholder data paths
# -----------------------------
LANDSCAPE_CSV = "path/to/LandscapeTable.csv"
SURGE_NPY = "path/to/SurgeData.npy"

# -----------------------------
# Load data (minimal assumption)
# -----------------------------
# LandscapeTable: one row per grid point
# Columns: lat, lon, canopy, manning, z0, ele, msl
landscape = pd.read_csv(
    LANDSCAPE_CSV,
    usecols=["lat", "lon", "canopy", "manning", "z0", "ele", "msl"]
)

# Surge array: shape (n_grid_points, n_storms)
surge = np.load(SURGE_NPY)

# -----------------------------
# Helper: train simple NN
# -----------------------------
def train_model(X_train, y_train):
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

    for _ in range(N_EPOCHS):
        optimizer.zero_grad()
        pred = model(X_tensor)
        loss = criterion(pred, y_tensor)
        loss.backward()
        optimizer.step()

    return model

# -----------------------------
# Main loop over NC
# -----------------------------
RMSE = []
R = []

for NC in NCs:
    print(f"Running NC = {NC}")

    # -------------------------
    # 1. K-means on spatial features
    # -------------------------
    geo_features = landscape[["lat", "lon"]].values

    kmeans = KMeans(
        n_clusters=NC,
        n_init=10,
        random_state=RANDOM_STATE
    )
    labels = kmeans.fit_predict(geo_features)
    centers = kmeans.cluster_centers_

    # Find closest grid point to each centroid
    centroid_indices = []
    for c in centers:
        idx = np.argmin(np.linalg.norm(geo_features - c, axis=1))
        centroid_indices.append(idx)

    centroid_indices = np.array(centroid_indices)

    # -------------------------
    # 2. Build training data
    # -------------------------
    X = landscape.iloc[centroid_indices].values
    y = surge[centroid_indices, :].mean(axis=1, keepdims=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Simple train / test split
    n_train = int(0.8 * len(X_scaled))
    X_train, X_test = X_scaled[:n_train], X_scaled[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # -------------------------
    # 3. Train model
    # -------------------------
    model = train_model(X_train, y_train)

    # -------------------------
    # 4. Test performance
    # -------------------------
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_pred = model(X_test_tensor).cpu().numpy()

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    corr = np.corrcoef(y_test.flatten(), y_pred.flatten())[0, 1]

    RMSE.append(rmse)
    R.append(corr)

    print(f"  RMSE = {rmse:.4f}, R = {corr:.4f}")

# -----------------------------
# Results summary
# -----------------------------
for nc, rmse, r in zip(NCs, RMSE, R):
    print(f"NC={nc:6d} | RMSE={rmse:.4f} | R={r:.4f}")
