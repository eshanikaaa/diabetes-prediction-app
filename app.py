import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.title("Diabetes Prediction App")

@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")

@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.pkl")

# Load files once
df = load_data()
model = load_model()

st.sidebar.header("Input Patient Data")

def user_input():
    pregnancies = st.sidebar.number_input('Pregnancies', 0, 20, 1)
    glucose = st.sidebar.number_input('Glucose', 0, 200, 100)
    bp = st.sidebar.number_input('Blood Pressure', 0, 150, 70)
    skin = st.sidebar.number_input('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.number_input('Insulin', 0, 900, 80)
    bmi = st.sidebar.number_input('BMI', 0.0, 70.0, 25.0)
    dpf = st.sidebar.number_input('Diabetes Pedigree Function', 0.0, 3.0, 0.5)
    age = st.sidebar.number_input('Age', 10, 100, 30)
    return pd.DataFrame([{
        'Pregnancies': pregnancies, 'Glucose': glucose, 'BloodPressure': bp,
        'SkinThickness': skin, 'Insulin': insulin, 'BMI': bmi,
        'DiabetesPedigreeFunction': dpf, 'Age': age
    }])

input_df = user_input()

if st.sidebar.button("Predict"):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    st.subheader("Prediction:")
    st.write("🟢 Not Diabetic" if pred == 0 else "🔴 Diabetic")
    st.subheader("Probability:")
    st.write(f"Not Diabetic: {proba[0]:.2f}, Diabetic: {proba[1]:.2f}")

    st.subheader("Feature Importance (Coefficients)")
    coeffs = model.coef_[0]
    features = input_df.columns
    fig, ax = plt.subplots()
    ax.barh(features, coeffs)
    ax.set_xlabel("Coefficient Value")
    st.pyplot(fig)

