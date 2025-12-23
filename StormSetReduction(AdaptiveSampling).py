"""
Minimal storm set reduction (Adaptive sampling algorithm) illustrative code
- Loads training data
- Trains surrogate once
- Runs ONE adaptive-learning iteration
- Selects next storm

This file is intended for public GitHub release.
No raw data is included.
"""

# =====================
# Imports
# =====================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

# =====================
# Placeholder paths (USER EDITS ONLY HERE)
# =====================
PROJECT_DIR = Path("path/to/project")

TRAINING_TABLE = PROJECT_DIR / "data/FullTrainingTable_example.csv"
X_TEST_FILE = PROJECT_DIR / "data/X_test_example.npy"
Y_TEST_FILE = PROJECT_DIR / "data/y_test_example.npy"

# =====================
# Configuration
# =====================
N_GRID = 80224          # number of spatial grid points
N_EPOCHS = 3
BATCH_SIZE = 1024
INITIAL_N_STORMS = 10

FEATURE_COLS = [
    'Heading', 'V_f (knots)', 'R_max (nm)',
    'Landfall Lon (x)', 'c_p (mbar)',
    'lat', 'lon', 'manning', 'ele', 'msl'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Model definition
# =====================
def build_model(input_dim=10):
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 1)
    ).to(device)

# =====================
# Load data
# =====================
print("Loading data...")

df = pd.read_csv(TRAINING_TABLE)

X = df[FEATURE_COLS].values
y = df[['surge']].values
storm_ids = df['Storm ID'].values

X_test = np.load(X_TEST_FILE)
y_test = np.load(Y_TEST_FILE)

# =====================
# Initial training set
# =====================
initial_storms = list(range(INITIAL_N_STORMS))

train_idx = [i for i, sid in enumerate(storm_ids) if sid in initial_storms]

X_train = X[train_idx]
y_train = y[train_idx]

# =====================
# Standardization
# =====================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# =====================
# Training
# =====================
print("Training surrogate model...")

model = build_model()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model.train()
for _ in range(N_EPOCHS):
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

# =====================
# Prediction on test data
# =====================
print("Running adaptive-learning step...")

X_test_scaled = scaler.transform(X_test)

model.eval()
with torch.no_grad():
    pred = model(
        torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    ).cpu().numpy()

# =====================
# Metrics
# =====================
rmse = np.sqrt(mean_squared_error(pred, y_test))
print(f"RMSE = {rmse:.4f}")

# =====================
# Adaptive-learning selection
# =====================
diff = (pred - y_test).reshape(N_GRID, -1, order='F')
energy = np.mean(diff ** 2, axis=0)

ranked = np.argsort(-energy)
new_storm = next(i for i in ranked if i not in initial_storms)

print(f"Selected next storm index: {new_storm}")
