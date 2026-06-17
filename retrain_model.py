"""
retrain_model.py
────────────────
Run this script ONCE from your MalariaProject folder:

    python retrain_model.py

It will:
  1. Load processed_malaria_dataset.csv
  2. Preprocess and build LSTM sequences
  3. Train the 2-layer LSTM for 100 epochs
  4. Save  malaria_lstm_final.pth   ← model weights
  5. Save  scaler_y_final.pkl       ← target scaler
  6. Print final RMSE / MAE / MAPE / R²

After this runs successfully, restart Streamlit and the dashboard
will work fully.
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────────────────────
CSV_PATH = "processed_malaria_dataset.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Cannot find {CSV_PATH}. "
        "Make sure you are running this script from your MalariaProject folder."
    )

df = pd.read_csv(CSV_PATH)
print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"   Columns: {list(df.columns)}")

# ─────────────────────────────────────────────────────────────
# 2.  PREPROCESS
# ─────────────────────────────────────────────────────────────
# Sort chronologically within each district
df = df.sort_values(["District","Year","Month_Num"]).reset_index(drop=True)

# Create lag features if not present
for col in ["Rainfall_lag1","Rainfall_lag2","Cases_lag1","Cases_lag2"]:
    if col not in df.columns:
        print(f"   Engineering {col} ...")

if "Rainfall_lag1" not in df.columns:
    df["Rainfall_lag1"] = df.groupby("District")["Rainfall_mm"].shift(1)
if "Rainfall_lag2" not in df.columns:
    df["Rainfall_lag2"] = df.groupby("District")["Rainfall_mm"].shift(2)
if "Cases_lag1" not in df.columns:
    df["Cases_lag1"] = df.groupby("District")["Confirmed_Malaria_Cases"].shift(1)
if "Cases_lag2" not in df.columns:
    df["Cases_lag2"] = df.groupby("District")["Confirmed_Malaria_Cases"].shift(2)

df = df.dropna().reset_index(drop=True)
print(f"✅ After lag engineering: {len(df)} rows")

# Features and target
FEATURES = ["Month_Num","Rainfall_mm","Temperature_C","RH_percent",
            "Rainfall_lag1","Rainfall_lag2","Cases_lag1","Cases_lag2"]
TARGET   = "Confirmed_Malaria_Cases"

X_raw = df[FEATURES].values.astype(np.float32)
y_raw = df[TARGET].values.astype(np.float32).reshape(-1, 1)

# Scale features
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_raw)

# Scale target — save this scaler
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y_raw)

# ─────────────────────────────────────────────────────────────
# 3.  SEQUENCE GENERATION  (window = 3 months)
# ─────────────────────────────────────────────────────────────
SEQ_LEN = 3

def make_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

X_seq, y_seq = make_sequences(X_scaled, y_scaled, SEQ_LEN)
print(f"✅ Sequences: X={X_seq.shape}, y={y_seq.shape}")

# Chronological 80/20 split
split = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train)
X_test_t  = torch.tensor(X_test)

# ─────────────────────────────────────────────────────────────
# 4.  MODEL
# ─────────────────────────────────────────────────────────────
class MalariaLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(8, 64, 2, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(64, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model     = MalariaLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ─────────────────────────────────────────────────────────────
# 5.  TRAINING
# ─────────────────────────────────────────────────────────────
EPOCHS = 100
print("\n🚀 Training LSTM model for 100 epochs...")
model.train()
for epoch in range(1, EPOCHS + 1):
    optimizer.zero_grad()
    output = model(X_train_t)
    loss   = criterion(output, y_train_t)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"   Epoch {epoch:3d}/100  —  Loss: {loss.item():.6f}")

# ─────────────────────────────────────────────────────────────
# 6.  EVALUATION
# ─────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    preds_scaled = model(X_test_t).numpy()

preds = scaler_y.inverse_transform(preds_scaled).flatten()
actuals = scaler_y.inverse_transform(y_test).flatten()

rmse = np.sqrt(mean_squared_error(actuals, preds))
mae  = mean_absolute_error(actuals, preds)
mape = np.mean(np.abs((actuals - preds) / (actuals + 1e-8))) * 100
r2   = r2_score(actuals, preds)

print(f"\n📊 Test Set Results:")
print(f"   RMSE : {rmse:.2f} cases")
print(f"   MAE  : {mae:.2f} cases")
print(f"   MAPE : {mape:.2f}%")
print(f"   R²   : {r2:.4f}")

# ─────────────────────────────────────────────────────────────
# 7.  SAVE FILES
# ─────────────────────────────────────────────────────────────
torch.save(model.state_dict(), "malaria_lstm_final.pth")
print("\n✅ Saved: malaria_lstm_final.pth")

with open("scaler_y_final.pkl", "wb") as f:
    pickle.dump(scaler_y, f)
print("✅ Saved: scaler_y_final.pkl")

print("\n🎉 Done! You can now run the Streamlit dashboard:")
print("   python -m streamlit run app.py")
