import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="🚗 CAR PRICE PREDICTION",
    page_icon="🚘",
    layout="wide"
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #001f3f 0%, #0074D9 100%);
            color: white;
        }
        .main-title {
            font-size: 3rem;
            color: #FFD700;
            text-align: center;
            font-weight: 800;
            text-shadow: 2px 2px 5px rgba(0,0,0,0.4);
            margin-bottom: 0.5rem;
        }
        .sub-title {
            text-align: center;
            font-size: 1.1rem;
            color: #f1f1f1;
            margin-bottom: 2rem;
        }
        .stButton button {
            background: linear-gradient(90deg, #FF6F61, #FFB400);
            color: white;
            border: none;
            padding: 0.8rem 2rem;
            border-radius: 12px;
            font-weight: 700;
            transition: all 0.3s ease-in-out;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
        }
        .stButton button:hover {
            transform: scale(1.07);
            background: linear-gradient(90deg, #FF8C00, #FFD700);
        }
        .css-1d391kg, .css-18e3th9 {
            background-color: rgba(255, 255, 255, 0.05);
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        .stSidebar {
            background: linear-gradient(180deg, #0A1F44, #004C99);
            color: white;
        }
        .stSidebar select, .stSidebar input {
            color: black !important;
        }
        .css-10trblm {
            color: #FFD700 !important;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Header Section
# -----------------------------
st.markdown("<h1 class='main-title'>🚗 CAR PRICE PREDICTION</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Estimate your car’s market value instantly Based on your Car features</p>", unsafe_allow_html=True)

# -----------------------------
# Load Trained Model
# -----------------------------
MODEL_PATH = "best_car_price_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found! Make sure 'best_car_price_model.pkl' exists in the same folder.")
    st.stop()

model = joblib.load(MODEL_PATH)

# -----------------------------
# Optional Image
# -----------------------------
if os.path.exists("car.png"):
    st.image("car.png", width=300, use_container_width=False)

# -----------------------------
# Sidebar Input Section
# -----------------------------
st.sidebar.markdown("## ⚙️ Input Car Details")

brand = st.sidebar.selectbox("Brand", 
    ['toyota','honda','hyundai','suzuki','bmw','audi','kia','mercedes','nissan','ford','other'])

car_age = st.sidebar.slider("Car Age (Years)", 0, 15, 5)
present_price = st.sidebar.number_input("Present Price (Lakhs)", 0.0, 50.0, 10.0, 0.5)
driven_kms = st.sidebar.number_input("Driven Kilometers", 0, 200000, 30000, 1000)
fuel_type = st.sidebar.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
transmission = st.sidebar.selectbox("Transmission", ['Manual', 'Automatic'])
selling_type = st.sidebar.selectbox("Selling Type", ['Dealer', 'Individual'])
owner = st.sidebar.selectbox("Owner Count", [0, 1, 2, 3])

# -----------------------------
# Create Input DataFrame
# -----------------------------
input_data = pd.DataFrame({
    'Present_Price': [present_price],
    'Driven_kms': [driven_kms],
    'Fuel_Type': [fuel_type],
    'Selling_type': [selling_type],
    'Transmission': [transmission],
    'Owner': [owner],
    'Car_Age': [car_age],
    'Brand': [brand]
})

# -----------------------------
# Prediction Section
# -----------------------------
st.markdown("---")
st.subheader("📊 Prediction Result")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🚀 Predict Car Price"):
        try:
            predicted_price = model.predict(input_data)[0]
            st.success(f"💰 **Estimated Selling Price:** ₹ {predicted_price:.2f} lakhs")
            st.balloons()
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")

# -----------------------------
# About Section
# -----------------------------
st.markdown("---")
st.markdown("""
<div style='text-align:center;'>
    <h3 style='color:#FFD700;'>📘 About This App</h3>
    <p style='color:#f2f2f2; font-size:1rem;'>
    This interactive web app predicts car prices using a trained Machine Learning model.  
    Enter your car details in the sidebar to get an accurate price estimate.
    </p>
</div>
""", unsafe_allow_html=True)

st.caption("👨‍💻 Developed by Sikander Ktk| Powered by Streamlit & Scikit-learn")
