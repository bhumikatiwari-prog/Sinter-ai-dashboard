import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np

st.set_page_config(page_title="Sinter AI Dashboard", layout="wide")

# This special Streamlit command ensures the AI only trains ONCE when the app loads
@st.cache_resource
def train_models():
    # 1. Load Beta Data
    beta_df = pd.read_csv('Beta.xlsx - Sheet1.csv', header=[0,1])
    beta_cols = [c[1] for c in beta_df.columns]
    beta_cols[0] = 'Date'
    beta_df.columns = beta_cols
    beta_df['Date'] = pd.to_datetime(beta_df['Date'], errors='coerce')
    beta_df = beta_df.dropna(subset=['Date'])
    for col in beta_df.columns:
        if col != 'Date': beta_df[col] = pd.to_numeric(beta_df[col], errors='coerce')

    # 2. Load Gamma Data
    with open('gamma.xlsx - May25-Mar26.csv', 'r') as f:
        lines = [next(f) for x in range(50)]
    header_idx = next(i for i, line in enumerate(lines) if 'Date' in line or 'Moisture' in line)
    gamma_df = pd.read_csv('gamma.xlsx - May25-Mar26.csv', skiprows=header_idx)
    gamma_df = gamma_df.loc[:, ~gamma_df.columns.str.contains('^Unnamed')]
    gamma_df.rename(columns={'Date ': 'Date', '%6.3MM(TI)': 'TI_gamma', 'RDI': 'RDI_gamma', 'RI': 'RI_gamma'}, inplace=True)
    gamma_df['Date'] = pd.to_datetime(gamma_df['Date'], errors='coerce')
    gamma_df = gamma_df.dropna(subset=['Date'])
    for col in gamma_df.columns:
        if col != 'Date': gamma_df[col] = pd.to_numeric(gamma_df[col], errors='coerce')

    # 3. Merge & Clean
    df = pd.merge(beta_df, gamma_df, on='Date', how='inner')
    df = df.drop(columns=['%MPS_y', 'TI_gamma', 'RDI_gamma', 'RI_gamma'], errors='ignore')
    df = df.rename(columns={'%MPS_x': '%MPS'})

    targets = ['TI', 'RDI', 'RI']
    features = [c for c in df.columns if c not in targets and c != 'Date']
    df[features] = df[features].fillna(df[features].mean())
    df[targets] = df[targets].fillna(df[targets].mean())

    X = df[features]
    
    # 4. Train AI Models
    models = {}
    for t in targets:
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, df[t])
        models[t] = rf
        
    return models, df[features].mean().to_dict(), features

# --- DASHBOARD UI ---
st.title("🔥 Sinter Quality Real-Time AI Predictor")
st.markdown("Adjust the process parameters below to see how they impact final sinter quality.")

with st.spinner("Reading plant data and training AI Model..."):
    models, feature_means, feature_names = train_models()

col1, col2 = st.columns(2)

with col1:
    st.subheader("🧪 Chemical Parameters")
    feo = st.slider("FeO (%)", 8.0, 14.0, 10.0, 0.1)
    basicity = st.slider("Basicity (B2)", 1.7, 2.5, 2.1, 0.01)
    sio2 = st.slider("SiO2 (%)", 4.0, 8.0, 6.0, 0.1)
    al2o3 = st.slider("Al2O3 (%)", 2.0, 5.0, 3.0, 0.1)

with col2:
    st.subheader("⚙️ Process Parameters")
    mc_speed = st.slider("Machine Speed (m/min)", 1.5, 3.5, 2.5, 0.1)
    moisture = st.slider("Moisture (%)", 4.0, 9.0, 6.0, 0.1)
    esp_temp = st.slider("ESP Inlet Temp (°C)", 120.0, 180.0, 145.0, 1.0)

# Build inputs
input_data = feature_means.copy()
input_data['%FeO'] = feo
input_data['Basicity  (B2)'] = basicity
input_data['%SiO2'] = sio2
input_data['%Al2O3'] = al2o3
input_data['M/c speed (m/min)'] = mc_speed
input_data['Moisture '] = moisture
input_data['ESP Inlet Temp. '] = esp_temp

input_df = pd.DataFrame([input_data], columns=feature_names)

# Predict
pred_TI = models['TI'].predict(input_df)[0]
pred_RDI = models['RDI'].predict(input_df)[0]
pred_RI = models['RI'].predict(input_df)[0]

st.divider()
st.subheader("🎯 Real-Time Predictions")

r1, r2, r3 = st.columns(3)

# Calculate how far the prediction is from your targets
delta_ti = pred_TI - 80.0
delta_rdi = pred_RDI - 21.0
delta_ri = pred_RI - 75.0

# For TI and RI, higher is better (normal). For RDI, lower is better (inverse).
r1.metric(label="Predicted TI (>80%)", value=f"{pred_TI:.2f}%", delta=f"{delta_ti:.2f}%", delta_color="normal")
r2.metric(label="Predicted RDI (<21%)", value=f"{pred_RDI:.2f}%", delta=f"{delta_rdi:.2f}%", delta_color="inverse")
r3.metric(label="Predicted RI (>75%)", value=f"{pred_RI:.2f}%", delta=f"{delta_ri:.2f}%", delta_color="normal")
