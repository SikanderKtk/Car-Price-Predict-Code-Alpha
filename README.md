# 🚗 Car Price Prediction

A machine learning project that predicts car prices based on key features such as brand, year, mileage, engine size, fuel type, and transmission. The project follows a complete data science pipeline — from data cleaning and preprocessing to model training and evaluation — and includes an interactive app for real-time predictions.

## 📌 Overview

This project was built as part of a Code Alpha internship task. It explores how vehicle attributes influence resale price and compares multiple regression models to find the best-performing one for accurate price prediction.

## ✨ Features

- End-to-end data science pipeline: data cleaning, preprocessing, feature engineering, model training, and evaluation
- Multiple regression models trained and compared: **Linear Regression**, **Decision Tree**, **Random Forest**, and **Gradient Boosting**
- Model evaluation using standard regression metrics, with results saved in `evaluation_summary.csv`
- Visualizations for model performance, including predicted vs. actual price plots and residual distribution analysis
- Best-performing model saved and ready for inference (`best_car_price_model.pkl`)
- Interactive web app (`app.py`) for predicting car prices from user input

## 🗂️ Project Structure

```
Car-Price-Predict-Code-Alpha/
│
├── CODE_ALPHA_TASK2.ipynb        # Main notebook: EDA, preprocessing, model training & evaluation
├── app.py                        # Interactive app for car price prediction
├── car_data.csv                  # Dataset used for training and evaluation
├── best_car_price_model.pkl      # Final selected best-performing model
├── linear_regression_model.pkl   # Trained Linear Regression model
├── random_forest_model.pkl       # Trained Random Forest model
├── gradient_boosting_model.pkl   # Trained Gradient Boosting model
├── evaluation_summary.csv        # Performance metrics comparison across models
├── predicted_vs_actual.png       # Visualization: predicted vs actual prices
├── residual_distribution.png     # Visualization: residual error distribution
├── requirements.txt              # Python dependencies
└── code alpha task 2 Report.docx # Written project report
```

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib/Seaborn
- **Modeling:** Linear Regression, Decision Tree, Random Forest, Gradient Boosting
- **App:** Streamlit (via `app.py`)

## ⚙️ How It Works

1. **Data Preprocessing** – Cleaned the raw dataset, handled missing values, and encoded categorical features (brand, fuel type, transmission, etc.)
2. **Exploratory Data Analysis** – Analyzed relationships between features like year, mileage, and engine size against price
3. **Model Training** – Trained and tuned multiple regression models on the processed data
4. **Model Evaluation** – Compared models using metrics such as R² score and error rates, logged in `evaluation_summary.csv`
5. **Model Selection** – Saved the best-performing model as `best_car_price_model.pkl`
6. **Deployment** – Built `app.py` so users can input car details and get an instant price prediction

## 🚀 Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/SikanderKtk/Car-Price-Predict-Code-Alpha.git
cd Car-Price-Predict-Code-Alpha
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

## 📊 Results

Model performance was evaluated and compared across all trained algorithms (see `evaluation_summary.csv`). Visualizations of predicted vs. actual prices and residual distributions are included to illustrate model accuracy and error patterns.

## 📄 Report

A detailed write-up of the methodology, findings, and results is available in `code alpha task 2 Report.docx`.

## 🙋 Author

**Muhammad Sikander Bakht**
Completed as part of a Data Science internship task at **Code Alpha**.

---
⭐ If you find this project useful, consider giving it a star!
