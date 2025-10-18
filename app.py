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
            color: black;
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
        .stSidebar {
            background: linear-gradient(180deg, #0A1F44, #004C99);
            color: white;
        }
        .stSidebar select, .stSidebar input {
            color: black !important;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Header Section
# -----------------------------
st.markdown("<h1 class='main-title'>🚗 CAR PRICE PREDICTION</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Estimate your car’s market value instantly based on your car features</p>", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = "best_car_price_model.pkl"
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found! Make sure 'best_car_price_model.pkl' exists in the same folder.")
    st.stop()

model = joblib.load(MODEL_PATH)

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
# Image and About Section (Side-by-side)
# -----------------------------
st.markdown("---")
col_img, col_about = st.columns([1, 1])

with col_img:
    if os.path.exists("car.png"):
        st.image("car.png", caption="Your Dream Car", use_container_width=True)

with col_about:
    st.markdown("""
<div style='text-align:center; font-size:1rem; color:black; line-height:1.7; background:rgba(255,255,255,0.05); padding:20px; border-radius:15px; box-shadow:0 4px 12px rgba(0,0,0,0.3); max-width:650px; margin:auto;'>
    <p>
        This interactive web app predicts <b style='color:#FFD700;'>car prices</b> using a trained 
        <b style='color:#FFD700;'>Machine Learning model</b>. <br><br>
        Enter your car details in the sidebar to get an accurate market price estimate instantly.  
        The model considers important features such as <b>brand, fuel type, transmission, car age,</b> 
        and <b>driven kilometers</b> to generate reliable results. <br><br>
        This project demonstrates a full <b style='color:#FFD700;'>end-to-end ML workflow</b> — 
        from <b>data preprocessing</b> to 
        <b>real-time prediction deployment</b> using Streamlit.
    </p>
</div>
""", unsafe_allow_html=True)


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
# Footer
# -----------------------------
st.markdown("---")
st.caption("👨‍💻 Developed by **Sikander Ktk** | Powered by **Streamlit & Scikit-learn**")
