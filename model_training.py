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

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3
})
os.makedirs("plots", exist_ok=True)

# SECTION 1 — CONFIGURATION

TIMESTEPS    = 3
HIDDEN_SIZE  = 64
NUM_LAYERS   = 2
EPOCHS       = 100
BATCH_SIZE   = 32
LEARNING_RATE = 0.001

print("=" * 50)
print("  MALARIA FORECASTING — LSTM MODEL TRAINING")
print("  SW Cameroon | 2015 – 2024")
print("=" * 50)

# SECTION 2 — LOAD PREPROCESSED DATA

df = pd.read_csv("processed_malaria_dataset.csv")

print(f"\n Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"  Districts : {sorted(df['District'].unique().tolist())}")

FEATURES = [
    "Month_Num",
    "Rainfall_mm",
    "Temperature_C",
    "RH_percent",
    "Rainfall_lag1",
    "Rainfall_lag2",
    "Cases_lag1",
    "Cases_lag2"
]
TARGET = "Confirmed_Malaria_Cases"

X = df[FEATURES].values.astype(np.float32)
y = df[TARGET].values.astype(np.float32).reshape(-1, 1)

print(f"  Features  : {len(FEATURES)}")
print(f"  Target    : {TARGET}")
print(f"  X shape   : {X.shape}")
print(f"  y range   : {y.min():.0f} – {y.max():.0f} cases")


# SECTION 3 — SCALE THE TARGET

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y)
pickle.dump(scaler_y, open("scaler_y.pkl", "wb"))
print("\n Target scaled to [0, 1] | scaler_y.pkl saved")


# SECTION 4 — CREATE SEQUENCES

def make_sequences(X, y, timesteps):
    Xs, ys = [], []
    for i in range(timesteps, len(X)):
        Xs.append(X[i - timesteps:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = make_sequences(X, y_scaled, TIMESTEPS)
print("\n Sequences created")
print(f"  X_seq shape : {X_seq.shape}  (samples, timesteps, features)")
print(f"  y_seq shape : {y_seq.shape}")


# SECTION 5 — TRAIN / TEST SPLIT (80/20)

split      = int(len(X_seq) * 0.80)
X_train    = X_seq[:split];   X_test  = X_seq[split:]
y_train    = y_seq[:split];   y_test  = y_seq[split:]

print("\n Train/test split (chronological)")
print(f"  Training samples : {len(X_train)}")
print(f"  Test samples     : {len(X_test)}")


# SECTION 6 — PYTORCH TENSORS & DATALOADER

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader  = DataLoader(train_dataset,
                           batch_size=BATCH_SIZE,
                           shuffle=False)

print("\n PyTorch tensors and DataLoader ready")
print(f"  Batches per epoch : {len(train_loader)}")

# SECTION 7 — DEFINE LSTM MODEL

class MalariaLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(MalariaLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = 0.2
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step   = lstm_out[:, -1, :]
        prediction  = self.fc(last_step)
        return prediction

INPUT_SIZE = X_train.shape[2]
model      = MalariaLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
criterion  = nn.MSELoss()
optimizer  = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

total_params = sum(p.numel() for p in model.parameters())
print("\n✓ LSTM model built")
print(f"  Input size    : {INPUT_SIZE} features")
print(f"  Hidden units  : {HIDDEN_SIZE}")
print(f"  LSTM layers   : {NUM_LAYERS}")
print(f"  Total params  : {total_params:,}")
print("  Loss function : MSELoss")
print(f"  Optimizer     : Adam (lr={LEARNING_RATE})")

# SECTION 8 — TRAINING LOOP

print(f"\n{'─'*50}")
print(f"  Training for {EPOCHS} epochs...")
print(f"{'─'*50}")

train_losses = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss   = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch [{epoch+1:3}/{EPOCHS}]  Loss: {avg_loss:.6f}")

torch.save(model.state_dict(), "malaria_lstm_model.pth")
print("\n Training complete | Model saved: malaria_lstm_model.pth")


# SECTION 9 — EVALUATION

model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_t).numpy()

y_pred   = scaler_y.inverse_transform(y_pred_scaled)
y_actual = scaler_y.inverse_transform(y_test)

rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
mae  = mean_absolute_error(y_actual, y_pred)
mape = np.mean(np.abs((y_actual - y_pred) / (y_actual + 1e-8))) * 100
ss_res = np.sum((y_actual - y_pred) ** 2)
ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
r2     = 1 - (ss_res / ss_tot)

print(f"\n{'='*50}")
print("  MODEL EVALUATION RESULTS")
print(f"{'='*50}")
print(f"  RMSE : {rmse:.2f}  (avg error in number of cases)")
print(f"  MAE  : {mae:.2f}  (mean absolute error)")
print(f"  MAPE : {mape:.2f}% (mean absolute percentage error)")
print(f"  R²   : {r2:.4f}  (1.0 = perfect fit)")
print(f"{'='*50}")


# SECTION 10 — PLOT

# Plot 1 — Training Loss Curve
plt.figure(figsize=(10, 4))
plt.plot(train_losses, color="#378ADD", linewidth=1.8, label="Training Loss")
plt.axhline(y=min(train_losses), color="#D85A30",
            linewidth=1, linestyle="--",
            label=f"Best loss: {min(train_losses):.6f}")
plt.title("LSTM Training Loss over Epochs", fontsize=13)
plt.xlabel("Epoch", fontsize=11)
plt.ylabel("MSE Loss", fontsize=11)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("plots/training_loss.png")
plt.show()
print("✓ Plot saved: plots/training_loss.png")

# Plot 2 — Predicted vs Actual
plt.figure(figsize=(13, 5))
plt.plot(y_actual, color="#1D9E75",
         linewidth=1.8, label="Actual cases")
plt.plot(y_pred, color="#D85A30",
         linewidth=1.8, linestyle="--", label="Predicted cases")
plt.fill_between(range(len(y_actual)),
                 y_actual.flatten(),
                 y_pred.flatten(),
                 alpha=0.1, color="#D85A30")
plt.title(f"LSTM Forecast vs Actual — Test Set   "
          f"(RMSE={rmse:.1f} | MAE={mae:.1f} | R²={r2:.3f})",
          fontsize=12)
plt.xlabel("Time Steps (months)", fontsize=11)
plt.ylabel("Confirmed Malaria Cases", fontsize=11)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("plots/predicted_vs_actual.png")
plt.show()
print("✓ Plot saved: plots/predicted_vs_actual.png")

# Plot 3 — Scatter: Actual vs Predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_actual, y_pred,
            alpha=0.5, color="#378ADD", s=25, edgecolors="none")
max_val = max(y_actual.max(), y_pred.max())
plt.plot([0, max_val], [0, max_val],
         color="#D85A30", linewidth=1.5,
         linestyle="--", label="Perfect prediction")
plt.title(f"Actual vs Predicted Cases   R²={r2:.3f}", fontsize=12)
plt.xlabel("Actual Cases", fontsize=11)
plt.ylabel("Predicted Cases", fontsize=11)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("plots/scatter_actual_vs_predicted.png")
plt.show()
print("✓ Plot saved: plots/scatter_actual_vs_predicted.png")

print("\n" + "="*50)
print("  ALL DONE — files saved in your folder:")
print("  malaria_lstm_model.pth")
print("  scaler_y.pkl")
print("  plots/training_loss.png")
print("  plots/predicted_vs_actual.png")
print("  plots/scatter_actual_vs_predicted.png")
print("="*50)
