import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Docker/headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import io
import os

# Page Config
st.set_page_config(page_title="Dengue Cases Predictor", layout="wide", page_icon="🦟")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main, [data-testid="stAppViewContainer"], .stApp {
        background-color: #f8f9fa !important;
    }
    .stButton>button, div.stButton > button {
        width: 100% !important;
        border-radius: 5px !important;
        height: 3em !important;
        background-color: #ff4b4b !important;
        color: white !important;
        font-weight: bold !important;
    }
    .risk-low { color: #28a745 !important; font-weight: bold !important; }
    .risk-medium { color: #ffc107 !important; font-weight: bold !important; }
    .risk-high { color: #dc3545 !important; font-weight: bold !important; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_and_train():
    model_path = "dengue_xgboost_model.pkl"
    
    # Check if a saved model exists
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                bundle = pickle.load(f)
            return (bundle['model'], bundle['le_province'], bundle['le_district'], 
                    bundle['features'], bundle['district_data'], bundle['weather_avg'], 
                    bundle['threshold_low'], bundle['threshold_high'])
        except Exception:
            # If loading fails, proceed to train
            pass

    # If no model found, train from scratch
    df = pd.read_csv('dengue_data_with_weather_data.csv')
    df = df.dropna(subset=['Cases'])
    
    # Fill missing weather data
    df['Temp_avg'] = df['Temp_avg'].interpolate(method='linear')
    df['Precipitation_avg'] = df['Precipitation_avg'].interpolate(method='linear')
    df['Humidity_avg'] = df['Humidity_avg'].interpolate(method='linear')
    df = df.fillna(df.median(numeric_only=True))

    # Encoders
    le_province = LabelEncoder()
    df['Province_Encoded'] = le_province.fit_transform(df['Province'])
    
    le_district = LabelEncoder()
    df['District_Encoded'] = le_district.fit_transform(df['District'])
    
    features = ['Year', 'Month', 'Province_Encoded', 'District_Encoded', 
                'Latitude', 'Longitude', 'Elevation', 
                'Temp_avg', 'Precipitation_avg', 'Humidity_avg']
    
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(df[features], df['Cases'])
    
    # Thresholds for risk
    threshold_low = df['Cases'].quantile(0.50)
    threshold_high = df['Cases'].quantile(0.85)
    
    # Metadata for district selection
    district_data = df[['District', 'Province', 'Latitude', 'Longitude', 'Elevation']].drop_duplicates().set_index('District').to_dict('index')
    
    # Historical weather averages per district and month
    weather_avg = df.groupby(['District', 'Month'])[['Temp_avg', 'Precipitation_avg', 'Humidity_avg']].mean().to_dict('index')
    
    # Save the full state bundle to disk
    model_bundle = {
        'model': model,
        'le_province': le_province,
        'le_district': le_district,
        'features': features,
        'district_data': district_data,
        'weather_avg': weather_avg,
        'threshold_low': threshold_low,
        'threshold_high': threshold_high
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)
        
    return model, le_province, le_district, features, district_data, weather_avg, threshold_low, threshold_high

# Load model and data
try:
    model, le_province, le_district, features, district_data, weather_avg, t_low, t_high = load_and_train()
except Exception as e:
    st.error(f"Error loading dependencies: {e}. Please ensure you have data and required libraries installed.")
    st.stop()

# --- Sidebar / Inputs ---
st.title("🦟 Dengue Case Prediction & Interpretation")
st.markdown("Predict the number of dengue cases and understand the factors driving the prediction.")

with st.sidebar:
    st.header("Input Parameters")
    district_list = sorted(list(district_data.keys()))
    selected_district = st.selectbox("Select District", district_list)
    
    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Year", min_value=2019, max_value=2030, value=2026)
    with col2:
        month = st.slider("Month", 1, 12, value=6)

    # Get defaults
    meta = district_data[selected_district]
    defaults = weather_avg.get((selected_district, month), 
                               {'Temp_avg': 27.0, 'Precipitation_avg': 1.0, 'Humidity_avg': 75.0})
    
    st.subheader("Weather Conditions")
    temp = st.slider("Average Temperature (°C)", 15.0, 40.0, float(defaults['Temp_avg']))
    precip = st.slider("Average Precipitation", 0.0, 15.0, float(defaults['Precipitation_avg']))
    humidity = st.slider("Average Humidity (%)", 30.0, 100.0, float(defaults['Humidity_avg']))

    st.divider()
    st.subheader("Model Export")
    # Bundle current state for download
    model_bundle = {
        'model': model,
        'le_province': le_province,
        'le_district': le_district,
        'features': features,
        'district_data': district_data,
        'weather_avg': weather_avg,
        'threshold_low': t_low,
        'threshold_high': t_high
    }
    
    # Save to buffer for download
    buffer = io.BytesIO()
    pickle.dump(model_bundle, buffer)
    
    st.download_button(
        label="📥 Download Model Bundle (.pkl)",
        data=buffer.getvalue(),
        file_name="dengue_xgboost_model.pkl",
        mime="application/octet-stream",
        help="Downloads a pickle file containing the trained model, label encoders, and feature list."
    )


# --- Prediction ---
if st.button("Predict Cases"):
    # Prepare input
    prov_enc = le_province.transform([meta['Province']])[0]
    dist_enc = le_district.transform([selected_district])[0]
    
    input_data = pd.DataFrame([[
        year, month, prov_enc, dist_enc,
        meta['Latitude'], meta['Longitude'], meta['Elevation'],
        temp, precip, humidity
    ]], columns=features)
    
    # Predict
    prediction = model.predict(input_data)[0]
    prediction = max(0, prediction) # No negative cases

    
    # Risk Level
    if prediction <= t_low:
        risk_text = "LOW"
        risk_class = "risk-low"
    elif prediction <= t_high:
        risk_text = "MEDIUM"
        risk_class = "risk-medium"
    else:
        risk_text = "HIGH"
        risk_class = "risk-high"
    
    # Display Results
    st.divider()
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.metric("Predicted Dengue Cases", f"{int(round(prediction))}")
        st.markdown(f"Risk Level: <span class='{risk_class}'>{risk_text}</span>", unsafe_allow_html=True)
    
    # --- SHAP Interpretation ---
    with res_col2:
        st.subheader("Explainability Panel")
        
        # Calculate SHAP for this single prediction
        explainer = shap.Explainer(model)
        shap_values = explainer(input_data)
        
        # Get feature contributions
        contribs = []
        for i, feat in enumerate(features):
            contribs.append((feat, shap_values.values[0][i]))
        
        # Sort by absolute strength
        contribs.sort(key=lambda x: abs(x[1]), reverse=True)
        
        top_pos = [f"**{f}**" for f, v in contribs if v > 0][:2]
        top_neg = [f"**{f}**" for f, v in contribs if v < 0][:2]
        
        explanation = f"The model predicts **{int(round(prediction))}** cases for **{selected_district}** in month **{month}**."
        
        if top_pos:
            explanation += f" This is primarily driven by an upward push from {', '.join(top_pos)}."
        if top_neg:
            explanation += f" However, {', '.join(top_neg)} helped in lowering the risk score."
        
        st.write(explanation)
        
        # Waterfall Plot
        fig, ax = plt.subplots()
        shap.waterfall_plot(shap_values[0], show=False)
        st.pyplot(fig)

    st.divider()
    st.info("Note: This model is for educational purposes and based on the historical dengue dataset from 2019-2021.")
