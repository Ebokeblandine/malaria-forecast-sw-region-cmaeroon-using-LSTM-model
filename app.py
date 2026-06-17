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
try:
    import importlib
    plotly = importlib.import_module('plotly')
    px = importlib.import_module('plotly.express')
    go = importlib.import_module('plotly.graph_objects')
    # import subplots dynamically to avoid static import errors in editors/linters
    try:
        subplots_mod = importlib.import_module('plotly.subplots')
        make_subplots = getattr(subplots_mod, 'make_subplots', None)
    except Exception:
        make_subplots = None
except Exception:
    px = None
    go = None
    make_subplots = None

# Ensure required plotting libs are available
if px is None or go is None or make_subplots is None:
    try:
        import streamlit as _st
        _st.error("Missing dependency: plotly is not installed. Please install plotly (pip install plotly) and restart the app.")
        _st.stop()
    except Exception:
        raise

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
FEATURES    = ["Month_Num","Rainfall_mm","Temperature_C","RH_percent",
               "Rainfall_lag1","Rainfall_lag2","Cases_lag1","Cases_lag2"]

RAW_MINS = {"Rainfall_mm":34.0,"Temperature_C":24.5,"RH_percent":71.1}
RAW_MAXS = {"Rainfall_mm":559.9,"Temperature_C":27.9,"RH_percent":93.0}

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
def scale_value(val, col):
    mn = RAW_MINS[col]; mx = RAW_MAXS[col]
    return (val - mn) / (mx - mn)

def get_risk_level(cases):
    if   cases < 400:  return "LOW",       "#27500A", "risk-low"
    elif cases < 600:  return "MODERATE",  "#633806", "risk-mod"
    elif cases < 900:  return "HIGH",      "#712B13", "risk-high"
    else:              return "VERY HIGH", "#791F1F", "risk-vhigh"

def risk_emoji(level):
    return {"LOW":"🟢","MODERATE":"🟡","HIGH":"🔴","VERY HIGH":"⛔"}.get(level,"")

def predict(model, scaler_y, df_proc, district,
            month_num, rainfall, temperature, humidity):
    dist_df = df_proc[df_proc["District"] == district].copy()
    if len(dist_df) < 2:
        return None
    last2 = dist_df.tail(2)[FEATURES].values.astype(np.float32)

    r_scaled  = scale_value(rainfall,    "Rainfall_mm")
    t_scaled  = scale_value(temperature, "Temperature_C")
    rh_scaled = scale_value(humidity,    "RH_percent")

    rain_lag1 = float(last2[-1, FEATURES.index("Rainfall_mm")])
    rain_lag2 = float(last2[-2, FEATURES.index("Rainfall_mm")])
    case_lag1 = float(last2[-1, FEATURES.index("Cases_lag1")])
    case_lag2 = float(last2[-2, FEATURES.index("Cases_lag1")])

    month_scaled = (month_num - 1) / 11.0
    new_row = np.array([[month_scaled, r_scaled, t_scaled, rh_scaled,
                         rain_lag1, rain_lag2, case_lag1, case_lag2]],
                       dtype=np.float32)

    seq   = np.concatenate([last2, new_row], axis=0)
    seq_t = torch.tensor(seq[np.newaxis, :, :], dtype=torch.float32)

    with torch.no_grad():
        pred_scaled = model(seq_t).numpy()
    pred = float(scaler_y.inverse_transform(pred_scaled)[0][0])
    return max(200, int(round(pred)))

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
         "📋 Forecast History"],
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
                st.rerun()
