# ============================================================
# MALARIA FORECASTING FLASK API
# ============================================================
#
# Project:
# Climate-Based Malaria Prediction System
# Using LSTM Deep Learning
#
# Dataset:
# South West Cameroon (2015 - 2024)
#
# Author:
# Blandine Ebole
#
# ============================================================

from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import pickle

# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

# ============================================================
# MODEL CONFIGURATION
# ============================================================

TIMESTEPS = 3

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

# ============================================================
# LSTM MODEL DEFINITION
# ============================================================

class MalariaLSTM(nn.Module):
    """
    Same architecture used during training.
    """

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=8,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Linear(64, 1)

    def forward(self, x):

        lstm_out, _ = self.lstm(x)

        last_step = lstm_out[:, -1, :]

        prediction = self.fc(last_step)

        return prediction


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("Loading trained model...")

model = MalariaLSTM()

model.load_state_dict(
    torch.load(
        "malaria_lstm_final.pth",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

print("Model loaded successfully")


# ============================================================
# LOAD TARGET SCALER
# ============================================================

print("Loading scaler...")

with open("scaler_y_final.pkl", "rb") as f:
    scaler_y = pickle.load(f)

print("Scaler loaded successfully")


# ============================================================
# ROOT ROUTE
# ============================================================

@app.route("/")
def home():

    return jsonify({
        "message": "Malaria Forecasting API",
        "status": "running"
    })


# ============================================================
# HEALTH CHECK ROUTE


@app.route("/health")
def health():

    return jsonify({
        "api_status": "healthy",
        "model_loaded": True
    })



# PREDICTION ROUTE
#
# Expected JSON:
#
# {
#   "sequence":[
#      [1,120,27,80,100,95,250,230],
#      [2,140,28,82,120,100,260,250],
#      [3,160,29,84,140,120,280,260]
#   ]
# }
#
# Shape:
# (3 timesteps × 8 features)
#
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

    
        # RECEIVE JSON DATA
        

        data = request.get_json()

        if "sequence" not in data:

            return jsonify({
                "error": "Missing 'sequence' field"
            }), 400

        sequence = np.array(
            data["sequence"],
            dtype=np.float32
        )

        
        # VALIDATE SHAPE
        

        if sequence.shape != (3, 8):

            return jsonify({
                "error":
                "Input must have shape (3,8). "
                "3 timesteps and 8 features."
            }), 400

        
        # RESHAPE FOR LSTM
        

        sequence = sequence.reshape(
            1,
            TIMESTEPS,
            len(FEATURES)
        )

        input_tensor = torch.tensor(
            sequence,
            dtype=torch.float32
        )

        
        # MODEL PREDICTION
        

        with torch.no_grad():

            prediction_scaled = model(
                input_tensor
            ).numpy()

        
        # CONVERT BACK TO REAL CASE NUMBERS
    

        prediction_actual = scaler_y.inverse_transform(
            prediction_scaled
        )

        predicted_cases = float(
            prediction_actual[0][0]
        )

        
        # RETURN RESULT
        

        return jsonify({

            "predicted_malaria_cases":
            round(predicted_cases, 2),

            "unit":
            "confirmed_cases"

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500



# RUN APPLICATION


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )