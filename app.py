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
    page_title="🚗 Car Price Prediction",
    page_icon="🚘",
    layout="centered"
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
    <style>
        .main-title {
            font-size: 2.5rem;
            color: #0072B5;
            text-align: center;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        .sub-title {
            text-align: center;
            font-size: 1.1rem;
            color: #444;
            margin-bottom: 2rem;
        }
        .stButton button {
            background: linear-gradient(90deg, #0072B5, #00B5AD);
            color: white;
            border: none;
            padding: 0.6rem 1.4rem;
            border-radius: 10px;
            font-weight: 600;
            transition: 0.3s;
        }
        .stButton button:hover {
            transform: scale(1.05);
            background: linear-gradient(90deg, #005f9e, #009f97);
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Header Section
# -----------------------------
st.markdown("<h1 class='main-title'>🚗 Car Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Predict your car's market value using Machine Learning.</p>", unsafe_allow_html=True)

# -----------------------------
# Load Trained Model
# -----------------------------
MODEL_PATH = "best_car_price_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found! Make sure 'best_car_price_model.pkl' exists in the same folder.")
    st.stop()

model = joblib.load(MODEL_PATH)

# -----------------------------
# Image (Optional)
# -----------------------------
if os.path.exists("car.png"):
    st.image("car.png", width=250)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.title("🔧 Input Features")

brand = st.sidebar.selectbox("Car Brand", 
    ['toyota','honda','hyundai','suzuki','bmw','audi','kia','mercedes','nissan','ford','other'])

car_age = st.sidebar.slider("Car Age (Years)", 0, 15, 5)
present_price = st.sidebar.number_input("Present Price (Lakhs)", 0.0, 50.0, 10.0, 0.5)
driven_kms = st.sidebar.number_input("Driven Kilometers", 0, 200000, 30000, 1000)
fuel_type = st.sidebar.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
transmission = st.sidebar.selectbox("Transmission Type", ['Manual', 'Automatic'])
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
# Predict Button
# -----------------------------
st.markdown("---")
st.subheader("📊 Prediction Result")

if st.button("🔍 Predict Car Price"):
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
st.info("""
**About:**  
This app uses a trained **Machine Learning model** to estimate car prices based on user-provided details.
Developed as a **Data Science capstone project**, it demonstrates model deployment on Streamlit.
""")

st.caption("👨‍💻 Built with Python by Sikander ktk | Streamlit | Scikit-learn")

