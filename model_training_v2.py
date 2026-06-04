import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle, os, warnings
warnings.filterwarnings("ignore")

os.makedirs("plots", exist_ok=True)

# ── CONFIGURATION ──────────────────────────────────────────
TIMESTEPS     = 6
HIDDEN_SIZE   = 64
NUM_LAYERS    = 2
EPOCHS        = 100
BATCH_SIZE    = 32
LEARNING_RATE = 0.001
DROPOUT       = 0.2

print("=" * 52)
print("  MALARIA LSTM FINAL — PER-DISTRICT SEQUENCES")
print("=" * 52)

# ── LOAD DATA ──────────────────────────────────────────────
df = pd.read_csv("processed_malaria_dataset.csv")
print(f"✓ Loaded: {df.shape[0]} rows")

FEATURES = [
    "Month_Num", "Rainfall_mm", "Temperature_C",
    "RH_percent", "Rainfall_lag1", "Rainfall_lag2",
    "Cases_lag1", "Cases_lag2"
]
TARGET = "Confirmed_Malaria_Cases"

# ── SCALE TARGET ───────────────────────────────────────────
scaler_y = MinMaxScaler()
df["target_scaled"] = scaler_y.fit_transform(
    df[[TARGET]].values)
pickle.dump(scaler_y, open("scaler_y_final.pkl", "wb"))
print("✓ Target scaled | scaler_y_final.pkl saved")

# ── CREATE SEQUENCES PER DISTRICT (the key fix) ────────────
def make_sequences(X, y, timesteps):
    Xs, ys = [], []
    for i in range(timesteps, len(X)):
        Xs.append(X[i - timesteps:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

all_X, all_y = [], []

for district in sorted(df["District"].unique()):
    sub = df[df["District"] == district].copy()
    X_d = sub[FEATURES].values.astype(np.float32)
    y_d = sub["target_scaled"].values.astype(
        np.float32).reshape(-1, 1)
    Xs, ys = make_sequences(X_d, y_d, TIMESTEPS)
    all_X.append(Xs)
    all_y.append(ys)
    print(f"  {district}: {len(Xs)} sequences created")

X_seq = np.concatenate(all_X, axis=0)
y_seq = np.concatenate(all_y, axis=0)

print(f"\n✓ Sequences: {X_seq.shape} "
      f"(samples, timesteps, features)")
print(f"  No district boundary contamination")

# ── 80/20 TRAIN/TEST SPLIT ─────────────────────────────────
split     = int(len(X_seq) * 0.80)
X_train   = X_seq[:split];  X_test  = X_seq[split:]
y_train   = y_seq[:split];  y_test  = y_seq[split:]

print(f"\n✓ Split — Train: {len(X_train)} | Test: {len(X_test)}")

# ── PYTORCH TENSORS ────────────────────────────────────────
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=BATCH_SIZE, shuffle=False)

# ── MODEL ──────────────────────────────────────────────────
class MalariaLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(MalariaLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=DROPOUT)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

INPUT_SIZE = X_train.shape[2]
model      = MalariaLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
criterion  = nn.MSELoss()
optimizer  = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE)

print(f"✓ Model: "
      f"{sum(p.numel() for p in model.parameters()):,} params")

# ── TRAINING ───────────────────────────────────────────────
print(f"\n  Training for {EPOCHS} epochs...")
train_losses = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for X_b, y_b in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg = epoch_loss / len(train_loader)
    train_losses.append(avg)
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch [{epoch+1:3}/{EPOCHS}]  "
              f"Loss: {avg:.6f}")

torch.save(model.state_dict(), "malaria_lstm_final.pth")
print("\n✓ Model saved: malaria_lstm_final.pth")

# ── EVALUATION ─────────────────────────────────────────────
model.eval()
with torch.no_grad():
    y_pred = scaler_y.inverse_transform(
        model(X_test_t).numpy())
y_actual = scaler_y.inverse_transform(y_test)

rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
mae  = mean_absolute_error(y_actual, y_pred)
mape = np.mean(np.abs((y_actual - y_pred)
               / (y_actual + 1e-8))) * 100
r2   = 1 - np.sum((y_actual - y_pred)**2) / \
       np.sum((y_actual - np.mean(y_actual))**2)

print(f"\n{'='*52}")
print(f"  FINAL RESULTS")
print(f"{'='*52}")
print(f"  RMSE : {rmse:.2f}")
print(f"  MAE  : {mae:.2f}")
print(f"  MAPE : {mape:.2f}%")
print(f"  R²   : {r2:.4f}")
print(f"  V1 : RMSE=57 | MAPE=7.16% | R²=0.9124")
print(f"{'='*52}")

# Save arrays for SHAP
np.save("X_train_shap.npy", X_train)
np.save("X_test_shap.npy",  X_test)
np.save("feature_names.npy", np.array(FEATURES))
print("\n✓ SHAP files saved — ready for next step")

# ── PLOTS ──────────────────────────────────────────────────
# Training loss
plt.figure(figsize=(10, 4))
plt.plot(train_losses, color="#378ADD", linewidth=1.8)
plt.title("Training Loss — LSTM Final")
plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
plt.tight_layout()
plt.savefig("plots/final_training_loss.png"); plt.show()
print("✓ plots/final_training_loss.png")

# Predicted vs actual
plt.figure(figsize=(13, 5))
plt.plot(y_actual, color="#1D9E75",
         linewidth=1.8, label="Actual")
plt.plot(y_pred, color="#D85A30",
         linewidth=1.8, linestyle="--", label="Predicted")
plt.fill_between(range(len(y_actual)),
                 y_actual.flatten(), y_pred.flatten(),
                 alpha=0.1, color="#D85A30")
plt.title(f"Forecast vs Actual  "
          f"RMSE={rmse:.1f} | MAPE={mape:.2f}% | R²={r2:.3f}")
plt.xlabel("Time Steps"); plt.ylabel("Confirmed Cases")
plt.legend(); plt.tight_layout()
plt.savefig("plots/final_predicted_vs_actual.png")
plt.show()
print("✓ plots/final_predicted_vs_actual.png")

# Scatter
plt.figure(figsize=(6, 6))
plt.scatter(y_actual, y_pred, alpha=0.5,
            color="#378ADD", s=25, edgecolors="none")
mx = max(y_actual.max(), y_pred.max())
plt.plot([0, mx], [0, mx], color="#D85A30",
         linestyle="--", label="Perfect prediction")
plt.title(f"Actual vs Predicted  R²={r2:.4f}")
plt.xlabel("Actual"); plt.ylabel("Predicted")
plt.legend(); plt.tight_layout()
plt.savefig("plots/final_scatter.png"); plt.show()
print("✓ plots/final_scatter.png")
print("\n  Done. Share your results and we move to SHAP.")