# 🚚 Delivery Time Prediction App

A Machine Learning–based web application to estimate food delivery time based on order details and system load conditions.

This project was built using XGBoost regression and deployed using Streamlit.

---

## 📌 Project Overview

Accurate delivery time estimation is crucial for customer satisfaction in food delivery services.  
This application predicts estimated delivery time (in minutes) using order-related features such as:

- Order information (market, protocol, category)
- Item details (quantity, price, subtotal)
- System load (busy partners, outstanding orders)
- Time-based features (hour, weekday, weekend)

---

## 🧠 Machine Learning Model

- Model: **XGBoost Regressor**
- Target: Delivery Time (minutes)
- Evaluation Metrics:
  - MAE: ~9–10 minutes
  - RMSE: ~12 minutes
  - R²: ~0.27 (baseline model)

Optimization techniques:
- Feature engineering
- One-hot encoding

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Joblib

---

## 📂 Project Structure

delivery_app/
│
├── app.py
├── model_final_project.pkl
├── final_features.pkl
├── requirements.txt
└── README.md

---

## 🚀 How to Run Locally

1. Clone repository
2. Create virtual environment
3. Install dependencies
4. Run Streamlit app

## 🌐 Deployment

This application is deployed using **Streamlit Cloud**.

---

## 📊 Future Improvements

- Add real-time traffic data
- Include distance-to-customer feature
- Improve model performance (increase R²)
- Add visual analytics dashboard
- Add prediction confidence interval

---

## 👤 Author

Ahmad Aldiyanto