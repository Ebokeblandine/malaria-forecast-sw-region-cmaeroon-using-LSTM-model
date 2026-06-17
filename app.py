import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sqlite3
import pickle
import os
from datetime import datetime
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# PAGE CONFIG
st.set_page_config(
    page_title="Malaria Forecast System — SW Cameroon",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# AUTHENTICATION SYSTEM
# ─────────────────────────────────────────────────────────────
import hashlib

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

def init_users_table():
    conn = sqlite3.connect("malaria_forecasts.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            full_name TEXT,
            role TEXT DEFAULT 'officer',
            created_at TEXT
        )
    """)
    hashed_pw = hashlib.sha256('admin123'.encode()).hexdigest()
    conn.execute("""
        INSERT OR IGNORE INTO users (username, password, full_name, role)
        VALUES ('admin', ?, 'Administrator', 'admin')
    """, (hashed_pw,))
    conn.commit()
    conn.close()

def authenticate(username, password):
    conn = sqlite3.connect("malaria_forecasts.db")
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    user = conn.execute("SELECT * FROM users WHERE username = ? AND password = ?",
                       (username, hashed_pw)).fetchone()
    conn.close()
    return user is not None

init_users_table()

# ── Login Screen ─────────────────────────────────────────────
if not st.session_state.logged_in:
    st.title("🦟 Malaria Forecast System")
    st.subheader("South West Region, Cameroon")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 🔐 Officer Login")
        username = st.text_input("Username", placeholder="admin")
        password = st.text_input("Password", type="password", placeholder="admin123")

        if st.button("Login", type="primary", use_container_width=True):
            if authenticate(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"✅ Welcome back, {username}!")
                st.rerun()
            else:
                st.error("❌ Incorrect username or password")
    st.stop()

st.sidebar.success(f"✅ Logged in as: **{st.session_state.username}**")

if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.session_state.username = None
    st.rerun()

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem; font-weight: 700;
        color: #2C5F2D; margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem; color: #6B7B6B; margin-bottom: 1.5rem;
    }
    .risk-card {
        padding: 1.2rem; border-radius: 12px;
        text-align: center; border: 1px solid #e0e0e0;
    }
    .risk-low    { background:#EAF3DE; border-color:#97BC62; }
    .risk-mod    { background:#FAEEDA; border-color:#BA7517; }
    .risk-high   { background:#FAECE7; border-color:#D85A30; }
    .risk-vhigh  { background:#FCEBEB; border-color:#E24B4A; }
    .metric-label { font-size:0.85rem; color:#6B7B6B; margin-bottom:0.3rem; }
    .metric-value { font-size:2rem; font-weight:700; color:#2C5F2D; }
    .metric-sub   { font-size:0.8rem; color:#6B7B6B; margin-top:0.2rem; }
    .section-header {
        font-size:1.3rem; font-weight:600; color:#2C5F2D;
        border-left:4px solid #2C5F2D; padding-left:0.7rem;
        margin:1.5rem 0 1rem;
    }
    .info-box {
        background:#E6F1FB; border-radius:8px;
        padding:1rem; border-left:4px solid #065A82;
        margin:1rem 0; font-size:0.9rem; color:#0C447C;
    }
    .stButton > button {
        background:#2C5F2D; color:white; border:none;
        border-radius:8px; padding:0.6rem 2rem;
        font-size:1rem; font-weight:600; width:100%;
    }
    .stButton > button:hover { background:#097A40; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
DISTRICTS   = ["Buea", "Limbe", "Muyuka", "Tiko", "Kumba"]
MONTHS      = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]
MONTH_MAP   = {m:i+1 for i,m in enumerate(MONTHS)}

# RAW (unscaled) column names — used to read from processed CSV
RAW_COLS = ["Month_Num","Rainfall_mm","Temperature_C","RH_percent",
            "Rainfall_lag1","Rainfall_lag2","Cases_lag1","Cases_lag2",
            "Confirmed_Malaria_Cases"]

# MinMax ranges from training data — used to scale inputs manually
RAW_MINS = {
    "Month_Num":      1.0,
    "Rainfall_mm":   34.0,
    "Temperature_C": 24.5,
    "RH_percent":    71.1,
    "Rainfall_lag1": 34.0,
    "Rainfall_lag2": 34.0,
    "Cases_lag1":   200.0,
    "Cases_lag2":   200.0,
}
RAW_MAXS = {
    "Month_Num":      12.0,
    "Rainfall_mm":   559.9,
    "Temperature_C":  27.9,
    "RH_percent":     93.0,
    "Rainfall_lag1": 559.9,
    "Rainfall_lag2": 559.9,
    "Cases_lag1":   1400.0,
    "Cases_lag2":   1400.0,
}

# Order the model expects
FEATURES = ["Month_Num","Rainfall_mm","Temperature_C","RH_percent",
            "Rainfall_lag1","Rainfall_lag2","Cases_lag1","Cases_lag2"]

DIST_COLORS = {
    "Buea":"#378ADD","Limbe":"#1D9E75","Muyuka":"#D85A30",
    "Tiko":"#7F77DD","Kumba":"#BA7517"
}

# ─────────────────────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────────────────────
def init_database():
    conn = sqlite3.connect("malaria_forecasts.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            district    TEXT    NOT NULL,
            month       INTEGER NOT NULL,
            month_name  TEXT    NOT NULL,
            year        INTEGER NOT NULL,
            rainfall    REAL    NOT NULL,
            temperature REAL    NOT NULL,
            humidity    REAL    NOT NULL,
            prediction  REAL    NOT NULL,
            risk_level  TEXT    NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_forecast(district, month_num, month_name, year,
                  rainfall, temperature, humidity,
                  prediction, risk_level):
    conn = sqlite3.connect("malaria_forecasts.db")
    conn.execute("""
        INSERT INTO forecasts
        (district, month, month_name, year, rainfall,
         temperature, humidity, prediction, risk_level)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (district, month_num, month_name, year,
          rainfall, temperature, humidity,
          prediction, risk_level))
    conn.commit()
    conn.close()

def load_forecast_history():
    conn = sqlite3.connect("malaria_forecasts.db")
    df = pd.read_sql_query(
        "SELECT * FROM forecasts ORDER BY created_at DESC LIMIT 100",
        conn)
    conn.close()
    return df

def clear_all_forecasts():
    conn = sqlite3.connect("malaria_forecasts.db")
    conn.execute("DELETE FROM forecasts")
    conn.commit()
    conn.close()

def create_new_user(username, password, full_name, role):
    try:
        conn = sqlite3.connect("malaria_forecasts.db")
        hashed_pw = hashlib.sha256(password.encode()).hexdigest()
        conn.execute("""
            INSERT INTO users (username, password, full_name, role, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (username, hashed_pw, full_name, role,
              datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False
    except Exception as e:
        st.error(f"Database error: {e}")
        return False

# ─────────────────────────────────────────────────────────────
# MODEL DEFINITION
# ─────────────────────────────────────────────────────────────
class MalariaLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(8, 64, 2,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ─────────────────────────────────────────────────────────────
# FILE LOADING  ← FIXED: using your actual filenames
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_scaler():
    model = MalariaLSTM()

    # ✅ FIXED: actual filename on disk
    model_path = "malaria_lstm_final.pth"
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: **{model_path}**  "
                 f"— run `retrain_model.py` first to regenerate it.")
        return None, None

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # ✅ FIXED: actual filename on disk
    scaler_path = "scaler_y_final.pkl"
    if not os.path.exists(scaler_path):
        st.error(f"❌ Scaler file not found: **{scaler_path}**  "
                 f"— run `retrain_model.py` first to regenerate it.")
        return None, None

    with open(scaler_path, "rb") as f:
        scaler_y = pickle.load(f)

    return model, scaler_y

@st.cache_data
def load_processed():
    # ✅ FIXED: actual filename on disk
    path = "processed_malaria_dataset.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

@st.cache_data
def load_raw():
    # ✅ FIXED: try processed dataset first (it has all the columns we need)
    for fname in ["processed_malaria_dataset.csv",
                  "malaria_SW_NASA_FINAL.csv",
                  "malaria_sw_cameroon_2015_2024_.csv"]:
        if os.path.exists(fname):
            return pd.read_csv(fname)
    return None

# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
def scale_col(val, col):
    """MinMax scale a single raw value using training-data ranges."""
    mn = RAW_MINS[col]; mx = RAW_MAXS[col]
    scaled = (val - mn) / (mx - mn)
    return float(np.clip(scaled, 0.0, 1.0))   # clamp to [0,1]

def build_scaled_row(month_num, rainfall, temperature, humidity,
                     rain_lag1, rain_lag2, case_lag1, case_lag2):
    """Return one scaled feature row in model input order."""
    return [
        scale_col(month_num,    "Month_Num"),
        scale_col(rainfall,     "Rainfall_mm"),
        scale_col(temperature,  "Temperature_C"),
        scale_col(humidity,     "RH_percent"),
        scale_col(rain_lag1,    "Rainfall_lag1"),
        scale_col(rain_lag2,    "Rainfall_lag2"),
        scale_col(case_lag1,    "Cases_lag1"),
        scale_col(case_lag2,    "Cases_lag2"),
    ]

def get_risk_level(cases):
    if   cases < 400:  return "LOW",       "#27500A", "risk-low"
    elif cases < 600:  return "MODERATE",  "#633806", "risk-mod"
    elif cases < 900:  return "HIGH",      "#712B13", "risk-high"
    else:              return "VERY HIGH", "#791F1F", "risk-vhigh"

def risk_emoji(level):
    return {"LOW":"🟢","MODERATE":"🟡","HIGH":"🔴","VERY HIGH":"⛔"}.get(level,"")

def predict(model, scaler_y, df_proc, district,
            month_num, rainfall, temperature, humidity):
    """
    Build a 3-month LSTM input sequence entirely from RAW values,
    scale every column correctly, then run inference.

    Sequence layout (3 rows × 8 features):
      row 0  = 2 months ago  (from processed CSV — raw columns)
      row 1  = 1 month ago   (from processed CSV — raw columns)
      row 2  = current month (user inputs + derived lags)
    """
    # ── Pull last 2 months of RAW data for this district ─────
    needed = ["Month_Num","Rainfall_mm","Temperature_C","RH_percent",
              "Confirmed_Malaria_Cases"]
    dist_df = df_proc[df_proc["District"] == district].copy()

    # Check required columns exist
    missing = [c for c in needed if c not in dist_df.columns]
    if missing:
        st.error(f"CSV is missing columns: {missing}. "
                 "Please upload the full processed_malaria_dataset.csv")
        return None

    if len(dist_df) < 2:
        return None

    dist_df = dist_df.sort_values(["Year","Month_Num"]).reset_index(drop=True)
    last2   = dist_df.tail(2)[needed].values   # shape (2, 5) — RAW values

    # Row indices
    r_minus2 = last2[0]   # 2 months ago: [Month_Num, Rain, Temp, RH, Cases]
    r_minus1 = last2[1]   # 1 month ago

    # Unpack raw values
    mn_2, rain_2, temp_2, rh_2, cases_2 = r_minus2
    mn_1, rain_1, temp_1, rh_1, cases_1 = r_minus1

    # ── Build 3 fully-scaled rows ─────────────────────────────
    # Row 0: 2 months ago
    # lag1 of that row = 3 months ago (unknown — use same-district mean as fallback)
    dist_mean_rain  = float(dist_df["Rainfall_mm"].mean())
    dist_mean_cases = float(dist_df["Confirmed_Malaria_Cases"].mean())

    row0 = build_scaled_row(
        month_num  = mn_2,
        rainfall   = rain_2,
        temperature= temp_2,
        humidity   = rh_2,
        rain_lag1  = dist_mean_rain,   # 3 months ago — use mean as proxy
        rain_lag2  = dist_mean_rain,
        case_lag1  = dist_mean_cases,
        case_lag2  = dist_mean_cases,
    )

    # Row 1: 1 month ago — lags are known from row 0
    row1 = build_scaled_row(
        month_num  = mn_1,
        rainfall   = rain_1,
        temperature= temp_1,
        humidity   = rh_1,
        rain_lag1  = rain_2,    # actual rainfall 2 months ago
        rain_lag2  = dist_mean_rain,
        case_lag1  = cases_2,   # actual cases 2 months ago
        case_lag2  = dist_mean_cases,
    )

    # Row 2: current user input — lags come from the two real months above
    row2 = build_scaled_row(
        month_num  = month_num,
        rainfall   = rainfall,
        temperature= temperature,
        humidity   = humidity,
        rain_lag1  = rain_1,    # last month's real rainfall
        rain_lag2  = rain_2,    # 2 months ago real rainfall
        case_lag1  = cases_1,   # last month's real cases
        case_lag2  = cases_2,   # 2 months ago real cases
    )

    # ── Stack into (1, 3, 8) tensor and run model ─────────────
    seq   = np.array([row0, row1, row2], dtype=np.float32)   # (3, 8)
    seq_t = torch.tensor(seq[np.newaxis, :, :])              # (1, 3, 8)

    with torch.no_grad():
        pred_scaled = model(seq_t).numpy()                   # (1, 1)

    pred = float(scaler_y.inverse_transform(pred_scaled)[0][0])
    # Remove the 200-floor so the model can output its true range
    return max(50, int(round(pred)))

# ─────────────────────────────────────────────────────────────
# INITIALISE
# ─────────────────────────────────────────────────────────────
init_database()
model, scaler_y = load_model_and_scaler()
df_proc = load_processed()
df_raw  = load_raw()

# ─────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦟 Malaria Forecast")
    st.markdown("**SW Cameroon Health Districts**")
    st.markdown("---")
    
    page = st.radio(
    "Navigate to:",
    ["🏠 Dashboard",
     "📊 Generate Forecast",
     "📈 Trends",
     "🔍 Model Insights",
     "📋 Forecast History",
     "🗃 Data Viewer",      # ← New page
     "⚙️ Admin"],
    label_visibility="collapsed"
)

    st.markdown("---")
    st.markdown("**Districts covered:**")
    for d in DISTRICTS:
        st.markdown(f"• {d}")
    st.markdown("---")
    st.caption(
        "NDOUKIE EBOKE BLANDINE\n"
        "FE22A254 | University of Buea\n"
        "2025/2026")

# ═════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD HOME
# ═════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown(
        '<div class="main-title">🦟 Malaria Forecast System</div>'
        '<div class="sub-title">Southwest Cameroon Health Districts'
        ' — Climate-Driven LSTM Early Warning</div>',
        unsafe_allow_html=True)

    st.markdown('<div class="section-header">Model Performance</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("RMSE","57 cases","Avg prediction error"),
        ("MAE", "45 cases","Mean abs error"),
        ("MAPE","7.16%",   "Excellent < 10%"),
        ("R²",  "0.9124",  "91% variance explained"),
    ]
    for col, (label, val, sub) in zip([c1,c2,c3,c4], metrics):
        with col:
            st.markdown(f"""
            <div class="risk-card risk-low" style="padding:1rem">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{val}</div>
              <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Current District Risk Overview</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Risk levels below are based on the most '
        'recent predictions stored in the database. Run a forecast for '
        'any district to update the risk cards.</div>',
        unsafe_allow_html=True)

    history = load_forecast_history()
    cols = st.columns(5)
    for col, dist in zip(cols, DISTRICTS):
        with col:
            dist_hist = (history[history["district"]==dist]
                         if len(history) > 0 else pd.DataFrame())
            if len(dist_hist) > 0:
                latest = dist_hist.iloc[0]
                pred   = int(latest["prediction"])
                level, tcolor, rclass = get_risk_level(pred)
            else:
                pred   = None
                level  = "No data"
                rclass = "risk-card"
                tcolor = "#6B7B6B"
            st.markdown(f"""
            <div class="risk-card {rclass}">
              <b style="color:#2C5F2D;font-size:1rem">{dist}</b><br>
              <span style="font-size:1.6rem;font-weight:700;color:{tcolor}">
                {risk_emoji(level)} {level}
              </span><br>
              <span style="font-size:0.85rem;color:#6B7B6B">
                {f"{pred:,} cases est." if pred else "Run a forecast →"}
              </span>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">About This System</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**What this system does:**
This system uses a trained Long Short-Term Memory (LSTM) deep
learning model to predict monthly malaria case counts in five
health districts of the Southwest Region of Cameroon.

It was trained on 10 years of monthly data (2015–2024) combining
malaria case records with real NASA POWER satellite climate data
— rainfall, temperature, and relative humidity.
        """)
    with c2:
        st.markdown("""
**How to use it:**
1. Go to **Generate Forecast** — select your district, enter the
   current month's climate values, and press Generate
2. The model predicts expected malaria cases for the coming month
3. A risk level (Low / Moderate / High / Very High) is assigned
4. All forecasts are saved automatically to the database
5. View past forecasts in **Forecast History**
        """)

# ═════════════════════════════════════════════════════════════
# PAGE 2 — GENERATE FORECAST
# ═════════════════════════════════════════════════════════════
elif page == "📊 Generate Forecast":
    st.markdown(
        '<div class="main-title">📊 Generate Malaria Forecast</div>'
        '<div class="sub-title">Enter current climate conditions to '
        'predict next month\'s malaria case count</div>',
        unsafe_allow_html=True)

    if model is None or df_proc is None:
        st.error("⚠️ Model or data files not found.")
        st.markdown("""
**To fix this, run the retraining script first:**
```
python retrain_model.py
```
This will regenerate:
- `malaria_lstm_final.pth`
- `scaler_y_final.pkl`
        """)
        st.stop()

    st.markdown('<div class="section-header">Input Parameters</div>',
                unsafe_allow_html=True)

    col_form, col_guide = st.columns([2, 1])

    with col_form:
        district = st.selectbox(
            "Health District", DISTRICTS,
            help="Select the district you want to forecast for")

        col1, col2 = st.columns(2)
        with col1:
            month_name = st.selectbox(
                "Forecast Month",
                MONTHS,
                index=datetime.now().month - 1,
                help="The month you want to predict cases for")
            year = st.number_input(
                "Year", min_value=2024, max_value=2035,
                value=datetime.now().year)
        with col2:
            rainfall = st.slider(
                "Rainfall this month (mm)",
                min_value=0, max_value=600,
                value=150,
                help="Total rainfall in millimetres for the current month")
            temperature = st.slider(
                "Mean temperature (°C)",
                min_value=22.0, max_value=30.0,
                value=26.0, step=0.1,
                help="Average temperature in degrees Celsius")
            humidity = st.slider(
                "Relative humidity (%)",
                min_value=60, max_value=100,
                value=80,
                help="Average relative humidity as a percentage")

        st.markdown("")
        run = st.button("🔮 Generate Forecast")

    with col_guide:
        st.markdown("""
**Typical SW Cameroon values:**

| Variable | Dry season | Wet season |
|---|---|---|
| Rainfall | 30–80 mm | 200–560 mm |
| Temperature | 26–28 °C | 24–26 °C |
| Humidity | 71–78 % | 85–93 % |

**Risk thresholds:**
- 🟢 **Low** — below 400 cases
- 🟡 **Moderate** — 400–600 cases
- 🔴 **High** — 600–900 cases
- ⛔ **Very High** — above 900 cases
        """)

    if run:
        month_num = MONTH_MAP[month_name]
        with st.spinner("Running LSTM model..."):
            prediction = predict(
                model, scaler_y, df_proc,
                district, month_num,
                float(rainfall), float(temperature), float(humidity))

        if prediction is None:
            st.error("Could not generate prediction. "
                     "Not enough historical data for this district.")
        else:
            level, tcolor, rclass = get_risk_level(prediction)
            save_forecast(
                district, month_num, month_name, year,
                rainfall, temperature, humidity,
                prediction, level)

            st.markdown("---")
            st.markdown(
                '<div class="section-header">Forecast Result</div>',
                unsafe_allow_html=True)

            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(f"""
                <div class="risk-card {rclass}">
                  <div class="metric-label">Predicted Cases</div>
                  <div class="metric-value" style="color:{tcolor}">
                    {prediction:,}
                  </div>
                  <div class="metric-sub">
                    {month_name} {year} · {district}
                  </div>
                </div>""", unsafe_allow_html=True)
            with r2:
                st.markdown(f"""
                <div class="risk-card {rclass}">
                  <div class="metric-label">Risk Level</div>
                  <div class="metric-value" style="color:{tcolor}">
                    {risk_emoji(level)} {level}
                  </div>
                  <div class="metric-sub">
                    Action required: {"YES" if level in ["HIGH","VERY HIGH"] else "MONITOR"}
                  </div>
                </div>""", unsafe_allow_html=True)
            with r3:
                st.markdown(f"""
                <div class="risk-card risk-low">
                  <div class="metric-label">Your Inputs</div>
                  <div class="metric-sub" style="text-align:left;padding:0.5rem 0">
                    🌧 Rainfall: {rainfall} mm<br>
                    🌡 Temperature: {temperature}°C<br>
                    💧 Humidity: {humidity}%<br>
                    📍 District: {district}
                  </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown(
                '<div class="section-header">Recommended Actions</div>',
                unsafe_allow_html=True)
            recs = {
                "LOW": [
                    "Continue routine malaria surveillance",
                    "Ensure adequate supply of ACTs and RDTs for baseline demand",
                    "Maintain community bed net distribution programme"
                ],
                "MODERATE": [
                    "Increase stock of artemisinin-based combination therapies (ACTs)",
                    "Alert community health workers to increase household visits",
                    "Distribute additional insecticide-treated nets in high-density areas"
                ],
                "HIGH": [
                    "URGENT: Pre-position emergency medicine stocks immediately",
                    "Deploy community health workers for active case finding",
                    "Issue district-level outbreak alert to Regional Delegation",
                    "Organise indoor residual spraying in high-burden sub-areas",
                    "Prepare additional beds at district health facility"
                ],
                "VERY HIGH": [
                    "CRITICAL ALERT: Notify Regional Delegation of Public Health NOW",
                    "Activate district emergency malaria response plan",
                    "Request emergency ACT and RDT supply from regional store",
                    "Mobilise all available community health workers immediately",
                    "Coordinate with neighbouring districts for resource sharing",
                    "Consider mass drug administration in highest-risk communities"
                ]
            }
            for rec in recs.get(level, []):
                icon = "🔴" if level in ["HIGH","VERY HIGH"] else "🟡"
                st.markdown(f"{icon} {rec}")

            st.success(f"✓ Forecast saved — {district}, {month_name} {year}")

# ═════════════════════════════════════════════════════════════
# PAGE 3 — TRENDS
# ═════════════════════════════════════════════════════════════
elif page == "📈 Trends":
    st.markdown(
        '<div class="main-title">📈 Historical Trends</div>'
        '<div class="sub-title">10-year malaria case and climate '
        'data for SW Cameroon districts</div>',
        unsafe_allow_html=True)

    if df_raw is None:
        st.error("⚠️ Dataset not found. Make sure **processed_malaria_dataset.csv** "
                 "is in your MalariaProject folder.")
        st.stop()

    # Build Month_Num if missing (raw CSV uses "Month" as name string)
    if "Month_Num" not in df_raw.columns:
        month_map_num = {m:i+1 for i,m in enumerate(MONTHS)}
        if "Month" in df_raw.columns:
            df_raw["Month_Num"] = df_raw["Month"].map(month_map_num)
        elif "Month_Name" in df_raw.columns:
            df_raw["Month_Num"] = df_raw["Month_Name"].map(month_map_num)

    df_raw = df_raw.sort_values(
        ["District","Year","Month_Num"]).reset_index(drop=True)

    sel_districts = st.multiselect(
        "Select districts to display:",
        DISTRICTS, default=DISTRICTS)

    df_view = df_raw[df_raw["District"].isin(sel_districts)].copy()
    df_view["Date"] = (df_view["Year"].astype(str) + "-" +
                       df_view["Month_Num"].astype(str).str.zfill(2))

    # ── Chart 1: Case trends ──────────────────────────────────
    st.markdown('<div class="section-header">'
                'Monthly Malaria Cases (2015–2024)</div>',
                unsafe_allow_html=True)
    fig1 = px.line(
        df_view, x="Date", y="Confirmed_Malaria_Cases",
        color="District",
        color_discrete_map=DIST_COLORS,
        labels={"Confirmed_Malaria_Cases":"Confirmed Cases","Date":"Month"},
        title="Monthly Confirmed Malaria Cases by District")
    fig1.update_layout(height=380, plot_bgcolor="white",
                       legend=dict(orientation="h", y=-0.2),
                       hovermode="x unified")
    fig1.update_xaxes(showgrid=False, nticks=20)
    fig1.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    st.plotly_chart(fig1, use_container_width=True)

    # ── Chart 2: Seasonal average ─────────────────────────────
    st.markdown('<div class="section-header">Average Seasonal Pattern</div>',
                unsafe_allow_html=True)
    seasonal = (df_view.groupby(["District","Month_Num"])
                ["Confirmed_Malaria_Cases"].mean().reset_index())
    fig2 = px.line(
        seasonal, x="Month_Num", y="Confirmed_Malaria_Cases",
        color="District",
        color_discrete_map=DIST_COLORS,
        markers=True,
        labels={"Confirmed_Malaria_Cases":"Avg Cases","Month_Num":"Month"},
        title="Average Monthly Malaria Seasonality")
    fig2.update_xaxes(tickvals=list(range(1,13)),
                      ticktext=[m[:3] for m in MONTHS],
                      showgrid=False)
    fig2.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    fig2.update_layout(height=350, plot_bgcolor="white",
                       legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig2, use_container_width=True)

    # ── Chart 3: Rainfall vs Cases (Buea) ────────────────────
    st.markdown('<div class="section-header">'
                'Rainfall vs Malaria Cases — Buea District</div>',
                unsafe_allow_html=True)
    buea = df_raw[df_raw["District"]=="Buea"].copy()
    buea["Date"] = (buea["Year"].astype(str) + "-" +
                    buea["Month_Num"].astype(str).str.zfill(2))
    fig3 = make_subplots(specs=[[{"secondary_y":True}]])
    fig3.add_trace(
        go.Scatter(x=buea["Date"], y=buea["Confirmed_Malaria_Cases"],
                   name="Confirmed Cases",
                   line=dict(color="#2C5F2D", width=2)),
        secondary_y=False)
    fig3.add_trace(
        go.Bar(x=buea["Date"], y=buea["Rainfall_mm"],
               name="Rainfall (mm)",
               marker_color="#B5D4F4", opacity=0.6),
        secondary_y=True)
    fig3.update_layout(
        title="Buea: Rainfall (bars) vs Malaria Cases (line)",
        height=380, plot_bgcolor="white",
        legend=dict(orientation="h", y=-0.2),
        hovermode="x unified")
    fig3.update_yaxes(title_text="Confirmed Cases", secondary_y=False)
    fig3.update_yaxes(title_text="Rainfall (mm)", secondary_y=True)
    st.plotly_chart(fig3, use_container_width=True)

    # ── Annual summary table ──────────────────────────────────
    st.markdown('<div class="section-header">Annual Summary</div>',
                unsafe_allow_html=True)
    annual = (df_view.groupby(["District","Year"])
              ["Confirmed_Malaria_Cases"]
              .agg(["sum","mean","max"])
              .round(0).astype(int).reset_index())
    annual.columns = ["District","Year","Total Cases","Avg Monthly","Peak Month"]
    st.dataframe(annual, use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════
# PAGE 4 — MODEL INSIGHTS
# ═════════════════════════════════════════════════════════════
elif page == "🔍 Model Insights":
    st.markdown(
        '<div class="main-title">🔍 Model Insights</div>'
        '<div class="sub-title">Understanding what drives the '
        'LSTM malaria forecast</div>',
        unsafe_allow_html=True)

    st.markdown('<div class="section-header">LSTM Model Architecture</div>',
                unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    arch = [
        ("Architecture","2-layer LSTM"),
        ("Hidden units","64 per layer"),
        ("Input features","8 features"),
        ("Look-back window","3 months"),
        ("Total parameters","52,289"),
    ]
    for col, (label, val) in zip([c1,c2,c3,c4,c5], arch):
        with col:
            st.markdown(f"""
            <div class="risk-card risk-low" style="padding:0.8rem">
              <div class="metric-label" style="font-size:0.8rem">{label}</div>
              <div style="font-size:1rem;font-weight:600;color:#2C5F2D">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Feature Importance Analysis</div>',
                unsafe_allow_html=True)

    # ── Built-in chart (always visible, no PNG needed) ────────
    features = ["Cases_lag1","Rainfall_lag1","Cases_lag2",
                "Rainfall_lag2","RH_percent","Temperature_C",
                "Rainfall_mm","Month_Num"]
    importance = [22.3, 18.7, 14.3, 11.2, 9.8, 8.4, 8.1, 7.2]
    colors_imp = ["#2C5F2D","#378ADD","#1D9E75","#7F77DD",
                  "#BA7517","#D85A30","#97BC62","#6B7B6B"]

    fig_imp = go.Figure(go.Bar(
        x=importance, y=features,
        orientation="h",
        marker_color=colors_imp,
        text=[f"{v}%" for v in importance],
        textposition="outside"))
    fig_imp.update_layout(
        title="Feature Importance — Permutation Analysis",
        xaxis_title="Importance (%)",
        height=350, plot_bgcolor="white",
        yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_imp, use_container_width=True)

    # ── PNG plots if they exist ───────────────────────────────
    plot_files = {
        "Feature Importance (Bar)":         "plots/feature_importance.png",
        "Lag Correlation":                   "plots/lag_correlation.png",
        "Two-Method Validation":             "plots/importance_two_methods.png",
        "Contribution Share (Donut)":        "plots/importance_donut.png",
    }
    available = {k:v for k,v in plot_files.items() if os.path.exists(v)}
    if available:
        tabs = st.tabs(list(available.keys()))
        for tab, fpath in zip(tabs, available.values()):
            with tab:
                st.image(fpath, use_column_width=True)

    st.markdown('<div class="section-header">Understanding the Lag Effect</div>',
                unsafe_allow_html=True)
    st.markdown("""
**Why does rainfall from last month predict this month's cases?**

When it rains heavily, mosquitoes lay eggs in standing water.
Larvae take **10–14 days** to develop into adults. Once a mosquito
bites an infected person, the parasite needs another **7–14 days**
inside the mosquito before it can infect someone new.

- Rain today → adult mosquitoes in ~2 weeks
- Infected mosquito → human cases in another ~2 weeks
- **Total delay: 4–8 weeks = 1–2 months at monthly resolution**

This is why **Rainfall_lag1** outperforms same-month rainfall.
The model discovered this biological pattern from 10 years of data.
    """)

    st.markdown('<div class="section-header">Training Performance</div>',
                unsafe_allow_html=True)
    train_plots = ["plots/training_loss.png",
                   "plots/scatter_actual_vs_predicted.png",
                   "plots/predicted_vs_actual.png"]
    found = [p for p in train_plots if os.path.exists(p)]
    if found:
        cols = st.columns(len(found))
        captions = ["Training loss","Actual vs Predicted","Forecast vs Actual"]
        for col, fpath, cap in zip(cols, found, captions):
            with col:
                st.image(fpath, caption=cap, use_column_width=True)
    else:
        st.info("Training plots not found in `plots/` folder. "
                "They will appear here once generated by your training script.")

# ═════════════════════════════════════════════════════════════
# PAGE 5 — FORECAST HISTORY
# ═════════════════════════════════════════════════════════════
elif page == "📋 Forecast History":
    st.markdown(
        '<div class="main-title">📋 Forecast History</div>'
        '<div class="sub-title">All forecasts generated and '
        'saved by health practitioners</div>',
        unsafe_allow_html=True)

    history = load_forecast_history()

    if len(history) == 0:
        st.info("No forecasts saved yet. Go to **Generate Forecast** "
                "to create your first prediction.")
    else:
        total     = len(history)
        high_risk = len(history[history["risk_level"].isin(["HIGH","VERY HIGH"])])
        districts_used = history["district"].nunique()
        latest_risk    = history.iloc[0]["risk_level"]
        level, tcolor, rclass = get_risk_level(history.iloc[0]["prediction"])

        c1,c2,c3,c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="risk-card risk-low">
              <div class="metric-label">Total Forecasts</div>
              <div class="metric-value">{total}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="risk-card risk-high">
              <div class="metric-label">High Risk Forecasts</div>
              <div class="metric-value" style="color:#D85A30">{high_risk}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="risk-card risk-low">
              <div class="metric-label">Districts Covered</div>
              <div class="metric-value">{districts_used}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="risk-card {rclass}">
              <div class="metric-label">Latest Risk</div>
              <div class="metric-value" style="color:{tcolor}">
                {risk_emoji(latest_risk)} {latest_risk}
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filter_dist = st.multiselect(
                "Filter by district:", DISTRICTS,
                default=list(history["district"].unique()))
        with col_f2:
            filter_risk = st.multiselect(
                "Filter by risk level:",
                ["LOW","MODERATE","HIGH","VERY HIGH"],
                default=["LOW","MODERATE","HIGH","VERY HIGH"])

        filtered = history[
            (history["district"].isin(filter_dist)) &
            (history["risk_level"].isin(filter_risk))
        ].copy()

        def color_risk(val):
            colors = {"LOW":"background-color:#EAF3DE",
                      "MODERATE":"background-color:#FAEEDA",
                      "HIGH":"background-color:#FAECE7",
                      "VERY HIGH":"background-color:#FCEBEB"}
            return colors.get(val,"")

        display_cols  = ["district","month_name","year","rainfall",
                         "temperature","humidity","prediction",
                         "risk_level","created_at"]
        display_names = ["District","Month","Year","Rainfall(mm)",
                         "Temp(°C)","Humidity(%)","Predicted Cases",
                         "Risk Level","Saved At"]

        disp = filtered[display_cols].copy()
        disp.columns = display_names

        st.dataframe(
            disp.style.map(color_risk, subset=["Risk Level"]),
            use_container_width=True, hide_index=True)

        if len(filtered) > 2:
            st.markdown('<div class="section-header">Forecast Trend</div>',
                        unsafe_allow_html=True)
            fig_hist = px.bar(
                filtered.sort_values("created_at"),
                x="created_at", y="prediction",
                color="risk_level",
                color_discrete_map={
                    "LOW":"#97BC62","MODERATE":"#BA7517",
                    "HIGH":"#D85A30","VERY HIGH":"#E24B4A"},
                facet_col="district", facet_col_wrap=3,
                labels={"prediction":"Predicted Cases",
                        "created_at":"Date Generated"},
                title="Forecast History by District and Risk Level")
            fig_hist.update_layout(height=400, plot_bgcolor="white")
            st.plotly_chart(fig_hist, use_container_width=True)

        col_dl, col_cl = st.columns([3,1])
        with col_dl:
            csv = filtered.to_csv(index=False)
            st.download_button(
                "⬇️ Download history as CSV",
                data=csv,
                file_name="malaria_forecast_history.csv",
                mime="text/csv")
        with col_cl:
            if st.button("🗑 Clear all forecasts"):
                clear_all_forecasts()
          
# ═════════════════════════════════════════════════════════════
# PAGE 6 — DATABASE VIEWER
# ═════════════════════════════════════════════════════════════
elif page == "🗃 Data Viewer":
    st.markdown('<div class="main-title">🗃 Database & Historical Data Viewer</div>', 
                unsafe_allow_html=True)
    st.subheader("South West Region Malaria + Climate Dataset (2015–2024)")

    # Load data
    df = load_raw()          # Try raw data first
    if df is None:
        df = load_processed()
    
    if df is None:
        st.error("No dataset found. Please place your CSV file in this folder.")
        st.stop()

    # Clean column names if needed
    df.columns = [col.strip() for col in df.columns]

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_districts = st.multiselect(
            "Select District(s)", 
            options=DISTRICTS, 
            default=DISTRICTS
        )
    with col2:
        selected_years = st.multiselect(
            "Select Year(s)", 
            options=sorted(df["Year"].unique()),
            default=sorted(df["Year"].unique())
        )
    with col3:
        months_available = sorted(df["Month"].unique()) if "Month" in df.columns else MONTHS
        selected_months = st.multiselect(
            "Select Month(s)", 
            options=months_available,
            default=months_available
        )

    # Filter the dataframe
    filtered_df = df[
        (df["District"].isin(selected_districts)) &
        (df["Year"].isin(selected_years))
    ]
    
    if "Month" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Month"].isin(selected_months)]

    # Summary Statistics
    st.markdown('<div class="section-header">📊 Summary Statistics</div>', unsafe_allow_html=True)
    summary = filtered_df.groupby("District").agg({
        "Confirmed_Malaria_Cases": ["count", "mean", "min", "max", "sum"]
    }).round(1)
    summary.columns = ["Records", "Avg Cases", "Min", "Max", "Total Cases"]
    st.dataframe(summary, use_container_width=True)

    # Main Data Table
    st.markdown('<div class="section-header">📋 Full Historical Data</div>', unsafe_allow_html=True)
    
    # Select columns to display
    display_cols = [col for col in ["District", "Year", "Month", "Confirmed_Malaria_Cases", 
                                  "Rainfall_mm", "Temperature_C", "RH_percent"] 
                   if col in filtered_df.columns]
    
    st.dataframe(
        filtered_df[display_cols].sort_values(["District", "Year", "Month"]),
        use_container_width=True,
        hide_index=True
    )

    # Download Button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Filtered Data as CSV",
        data=csv,
        file_name="malaria_filtered_data.csv",
        mime="text/csv",
        use_container_width=True
    )

    # Quick Visualization
    st.markdown('<div class="section-header">📈 Cases Trend (Filtered)</div>', unsafe_allow_html=True)
    if len(filtered_df) > 0:
        fig = px.line(
            filtered_df.sort_values(["District","Year","Month"]),
            x="Year", 
            y="Confirmed_Malaria_Cases",
            color="District",
            markers=True,
            title="Malaria Cases Trend by District (Filtered View)"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data matches the current filters.")
        
        # ═════════════════════════════════════════════════════════════
# PAGE — ADMIN DASHBOARD
# ═════════════════════════════════════════════════════════════
elif page == "⚙️ Admin":
    st.markdown('<div class="main-title">⚙️ Admin Dashboard</div>', unsafe_allow_html=True)
    st.subheader("System Management")

    if st.session_state.username != "admin":
        st.error("🔒 Only the Administrator can access this page.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["👤 User Management", "📊 System Overview", "🗑️ Maintenance"])

    # TAB 1: User Management
    with tab1:
        st.subheader("Create New Health Officer Account")
        
        with st.form("new_user_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                new_username = st.text_input("Username")
                new_fullname = st.text_input("Full Name", value="Health Officer")
            with col2:
                new_password = st.text_input("Password", type="password")
                new_role = st.selectbox("Role", ["officer", "admin"])
            
            submitted = st.form_submit_button("✅ Create User")
            
            if submitted:
                if new_username and new_password:
                    if create_new_user(new_username, new_password, new_fullname, new_role):
                        st.success(f"✅ User **{new_username}** created successfully!")
                    else:
                        st.error("❌ Username already exists. Choose another.")
                else:
                    st.warning("Please fill in all required fields.")

        st.markdown("---")
        st.subheader("Existing Users")
        conn = sqlite3.connect("malaria_forecasts.db")
        users_df = pd.read_sql_query(
            "SELECT username, full_name, role, created_at FROM users ORDER BY created_at DESC", 
            conn
        )
        conn.close()
        
        if not users_df.empty:
            st.dataframe(users_df, use_container_width=True, hide_index=True)
        else:
            st.info("No users found.")

    # TAB 2: System Overview
    with tab2:
        st.subheader("System Information")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            total_forecasts = len(load_forecast_history())
            st.metric("Total Forecasts", total_forecasts)
        with col2:
            conn = sqlite3.connect("malaria_forecasts.db")
            total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            conn.close()
            st.metric("Registered Users", total_users)
        with col3:
            st.metric("Districts Covered", len(DISTRICTS))

        st.markdown("---")
        st.subheader("Recent Forecasts")
        recent = load_forecast_history().head(10)
        if not recent.empty:
            st.dataframe(recent[["district", "month_name", "year", "prediction", "risk_level", "created_at"]], 
                        use_container_width=True, hide_index=True)
        else:
            st.info("No forecasts yet.")

    # TAB 3: Maintenance
    with tab3:
        st.subheader("Database Maintenance")
        
        if st.button("🗑️ Clear All Forecasts", type="secondary"):
            if st.checkbox("I am sure I want to delete ALL forecasts"):
                clear_all_forecasts()
                st.success("All forecasts have been cleared.")
                st.rerun()

        st.warning("⚠️ This action cannot be undone.")