import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="Flood Risk Prediction System",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Flood Risk Category Classification System")
st.markdown("Enter pre-disaster environmental indicators to evaluate real-time flood risk probabilities.")

# Load Model Artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_artifacts()
    st.sidebar.success("Model & Scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# Sidebar / Main Form Input Features
st.subheader("📊 Input Environmental Variables")

col1, col2 = st.columns(2)

with col1:
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=30000.0, value=1200.0, step=10.0)
    elevation = st.number_input("Elevation (m)", min_value=-50.0, max_value=8000.0, value=250.0, step=5.0)
    slope = st.number_input("Slope (degrees)", min_value=0.0, max_value=90.0, value=5.0, step=0.1)
    distance = st.number_input("Distance to River/Waterbody (m)", min_value=0.0, max_value=2000000.0, value=1000.0, step=50.0)

with col2:
    latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=23.7, step=0.01)
    longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=90.4, step=0.01)
    duration = st.number_input("Event Duration (days/hours)", min_value=0.0, max_value=1000.0, value=0.0, step=1.0)
    time_yr = st.number_input("Year / Time Indicator", min_value=1900, max_value=2030, value=2024, step=1)

# Prediction Logic
if st.button("🚀 Predict Flood Risk", type="primary"):
    # Construct input dataframe
    input_data = pd.DataFrame([{
        'Latitude': latitude,
        'Longitude': longitude,
        'duration': duration,
        'time': time_yr,
        'Rainfall': rainfall,
        'Elevation': elevation,
        'Slope': slope,
        'distance': distance
    }])
    
    # Scale inputs
    scaled_input = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]
    flood_prob = probabilities[1] * 100

    st.markdown("---")
    st.subheader("🎯 Prediction Results")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        if prediction == 1:
            st.error("🚨 **HIGH FLOOD RISK DETECTED**")
        else:
            st.success("✅ **LOW / NO FLOOD RISK DETECTED**")
            
    with col_b:
        st.metric(label="Estimated Flood Probability", value=f"{flood_prob:.2f}%")
        st.progress(int(flood_prob))

    # Feature Insights
    st.markdown("---")
    st.subheader("💡 Environmental Risk Breakdown")
    st.write(f"- **Rainfall Input:** {rainfall} mm (Primary contributor to risk profile)")
    st.write(f"- **Terrain Elevation:** {elevation} m")
    st.write(f"- **Terrain Slope:** {slope}°")