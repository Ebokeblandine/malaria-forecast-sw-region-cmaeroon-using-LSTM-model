"""
Malaria Forecast Decision Support System
SW Cameroon — University of Buea
Run: streamlit run app.py
"""

import streamlit as st
import torch, torch.nn as nn
import numpy as np, pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3, pickle, os
from datetime import datetime
from auth import init_users_db, show_login_page, show_add_user_page

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Malaria Forecast System — SW Cameroon",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# FIX: FIND MODEL FILES IN MULTIPLE LOCATIONS
# ─────────────────────────────────────────────────────────────
def find_file(filename):
    """Search for a file in common locations."""
    # 1. Same folder as app.py
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base, filename),
        os.path.join(base, "model", filename),
        os.path.join(base, "models", filename),
        filename,  # current working directory
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

MODEL_PATH   = find_file("malaria_lstm_model.pth")
SCALER_PATH  = find_file("scaler_y.pkl")
PROC_PATH    = find_file("processed_malaria_dataset.csv")
RAW_PATH     = (find_file("malaria_SW_NASA_FINAL.csv") or
                find_file("malaria_sw_cameroon_2015_2024_.csv"))

# ─────────────────────────────────────────────────────────────
# INITIALISE DATABASE + AUTH
# ─────────────────────────────────────────────────────────────
init_users_db()

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None

# ─────────────────────────────────────────────────────────────
# SHOW LOGIN IF NOT AUTHENTICATED
# ─────────────────────────────────────────────────────────────
if not st.session_state["logged_in"]:
    show_login_page()
    st.stop()   # stop rendering rest of app until logged in

# ─────────────────────────────────────────────────────────────
# FROM HERE: USER IS LOGGED IN
# ─────────────────────────────────────────────────────────────
current_user = st.session_state["user"]

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-title {font-size:2rem;font-weight:700;color:#2C5F2D;margin-bottom:0.2rem}
.sub-title  {font-size:1rem;color:#6B7B6B;margin-bottom:1.5rem}
.risk-card  {padding:1.2rem;border-radius:12px;text-align:center;border:1px solid #e0e0e0}
.risk-low   {background:#EAF3DE;border-color:#97BC62}
.risk-mod   {background:#FAEEDA;border-color:#BA7517}
.risk-high  {background:#FAECE7;border-color:#D85A30}
.risk-vhigh {background:#FCEBEB;border-color:#E24B4A}
.metric-label{font-size:0.85rem;color:#6B7B6B;margin-bottom:0.3rem}
.metric-value{font-size:2rem;font-weight:700;color:#2C5F2D}
.metric-sub  {font-size:0.8rem;color:#6B7B6B;margin-top:0.2rem}
.section-header{font-size:1.3rem;font-weight:600;color:#2C5F2D;
    border-left:4px solid #2C5F2D;padding-left:0.7rem;margin:1.5rem 0 1rem}
.info-box{background:#E6F1FB;border-radius:8px;padding:1rem;
    border-left:4px solid #065A82;margin:1rem 0;font-size:0.9rem;color:#0C447C}
.stButton > button{background:#2C5F2D;color:white;border:none;
    border-radius:8px;padding:0.6rem 2rem;font-size:1rem;
    font-weight:600;width:100%}
.stButton > button:hover{background:#097A40}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
DISTRICTS  = ["Buea","Limbe","Muyuka","Tiko","Kumba"]
MONTHS     = ["January","February","March","April","May","June",
              "July","August","September","October","November","December"]
MONTH_MAP  = {m:i+1 for i,m in enumerate(MONTHS)}
FEATURES   = ["Month_Num","Rainfall_mm","Temperature_C","RH_percent",
              "Rainfall_lag1","Rainfall_lag2","Cases_lag1","Cases_lag2"]
RAW_MINS   = {"Rainfall_mm":34.0,"Temperature_C":24.5,"RH_percent":71.1}
RAW_MAXS   = {"Rainfall_mm":559.9,"Temperature_C":27.9,"RH_percent":93.0}
DIST_COLORS= {"Buea":"#378ADD","Limbe":"#1D9E75","Muyuka":"#D85A30",
              "Tiko":"#7F77DD","Kumba":"#BA7517"}

# ─────────────────────────────────────────────────────────────
# DATABASE HELPERS
# ─────────────────────────────────────────────────────────────
def init_forecasts_db():
    conn = sqlite3.connect("malaria_forecasts.db")
    conn.execute("""CREATE TABLE IF NOT EXISTS forecasts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        district TEXT, month INTEGER, month_name TEXT,
        year INTEGER, rainfall REAL, temperature REAL,
        humidity REAL, prediction REAL, risk_level TEXT,
        created_by TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()

def save_forecast(district, month_num, month_name, year,
                  rainfall, temperature, humidity,
                  prediction, risk_level, username):
    conn = sqlite3.connect("malaria_forecasts.db")
    conn.execute("""INSERT INTO forecasts
        (district,month,month_name,year,rainfall,temperature,
         humidity,prediction,risk_level,created_by)
        VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (district,month_num,month_name,year,rainfall,
         temperature,humidity,prediction,risk_level,username))
    conn.commit(); conn.close()

def load_forecast_history():
    conn = sqlite3.connect("malaria_forecasts.db")
    df = pd.read_sql_query(
        "SELECT * FROM forecasts ORDER BY created_at DESC LIMIT 200",conn)
    conn.close(); return df

def clear_all_forecasts():
    conn = sqlite3.connect("malaria_forecasts.db")
    conn.execute("DELETE FROM forecasts")
    conn.commit(); conn.close()

init_forecasts_db()

# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────
class MalariaLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(8,64,2,batch_first=True,dropout=0.2)
        self.fc   = nn.Linear(64,1)
    def forward(self,x):
        out,_ = self.lstm(x); return self.fc(out[:,-1,:])

@st.cache_resource
def load_model_and_scaler():
    if not MODEL_PATH:
        return None, None
    if not SCALER_PATH:
        return None, None
    model = MalariaLSTM()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    with open(SCALER_PATH,"rb") as f:
        scaler_y = pickle.load(f)
    return model, scaler_y

@st.cache_data
def load_processed():
    return pd.read_csv(PROC_PATH) if PROC_PATH else None

@st.cache_data
def load_raw():
    return pd.read_csv(RAW_PATH) if RAW_PATH else None

model, scaler_y = load_model_and_scaler()
df_proc = load_processed()
df_raw  = load_raw()

# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
def scale_value(val, col):
    mn = RAW_MINS[col]; mx = RAW_MAXS[col]
    return (val - mn) / (mx - mn)

def get_risk_level(cases):
    if   cases < 400: return "LOW",       "#27500A","risk-low"
    elif cases < 600: return "MODERATE",  "#633806","risk-mod"
    elif cases < 900: return "HIGH",      "#712B13","risk-high"
    else:             return "VERY HIGH", "#791F1F","risk-vhigh"

def risk_emoji(level):
    return {"LOW":"🟢","MODERATE":"🟡","HIGH":"🔴","VERY HIGH":"⛔"}.get(level,"")

def predict(model, scaler_y, df_proc, district,
            month_num, rainfall, temperature, humidity):
    dist_df = df_proc[df_proc["District"]==district].copy()
    if len(dist_df) < 2: return None
    last2 = dist_df.tail(2)[FEATURES].values.astype(np.float32)
    r_sc  = scale_value(rainfall,"Rainfall_mm")
    t_sc  = scale_value(temperature,"Temperature_C")
    rh_sc = scale_value(humidity,"RH_percent")
    rain_lag1 = float(last2[-1, FEATURES.index("Rainfall_mm")])
    rain_lag2 = float(last2[-2, FEATURES.index("Rainfall_mm")])
    case_lag1 = float(last2[-1, FEATURES.index("Cases_lag1")])
    case_lag2 = float(last2[-2, FEATURES.index("Cases_lag1")])
    month_sc  = (month_num - 1) / 11.0
    new_row   = np.array([[month_sc, r_sc, t_sc, rh_sc,
                            rain_lag1, rain_lag2,
                            case_lag1, case_lag2]],
                          dtype=np.float32)
    seq   = np.concatenate([last2, new_row], axis=0)
    seq_t = torch.tensor(seq[np.newaxis,:,:], dtype=torch.float32)
    with torch.no_grad():
        pred_s = model(seq_t).numpy()
    return max(200, int(round(float(
        scaler_y.inverse_transform(pred_s)[0][0]))))

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦟 Malaria Forecast")
    st.markdown("**SW Cameroon Health Districts**")
    st.markdown(f"👤 **{current_user['full_name']}**")
    st.caption(f"Role: {current_user['role'].replace('_',' ').title()}")
    if current_user.get("district"):
        st.caption(f"District: {current_user['district']}")
    st.markdown("---")

    nav_options = ["🏠 Dashboard",
                   "📊 Generate Forecast",
                   "📈 Trends",
                   "🔍 Model Insights",
                   "📋 Forecast History"]
    if current_user["role"] == "admin":
        nav_options.append("👥 User Management")

    page = st.radio("Navigate to:", nav_options,
                    label_visibility="collapsed")
    st.markdown("---")
    if st.button("🚪 Logout"):
        st.session_state["logged_in"] = False
        st.session_state["user"]      = None
        st.rerun()
    st.markdown("---")
    st.caption("University of Buea\nNDOUKIE EBOKE BLANDINE\nFE22A254")

# ─────────────────────────────────────────────────────────────
# MODEL STATUS WARNING
# ─────────────────────────────────────────────────────────────
if not MODEL_PATH or not SCALER_PATH or not PROC_PATH:
    missing = []
    if not MODEL_PATH:   missing.append("malaria_lstm_model.pth")
    if not SCALER_PATH:  missing.append("scaler_y.pkl")
    if not PROC_PATH:    missing.append("processed_malaria_dataset.csv")
    st.error(
        f"⚠️ Missing file(s): **{', '.join(missing)}**  \n"
        f"Make sure these files are in the **same folder** as app.py.  \n"
        f"Current folder: `{os.path.dirname(os.path.abspath(__file__))}`")

# ─────────────────────────────────────────────────────────────
# USER MANAGEMENT PAGE
# ─────────────────────────────────────────────────────────────
if page == "👥 User Management":
    show_add_user_page(current_user)
    st.stop()

# ─────────────────────────────────────────────────────────────
# ALL OTHER PAGES — paste your existing page code below here
# (Dashboard, Forecast, Trends, Insights, History)
# ─────────────────────────────────────────────────────────────
