import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load('diabetes_model.pkl')

# Page title
st.title('Diabetes Prediction App')

# Sidebar input
st.sidebar.header('Input Patient Data')

def user_input():
    pregnancies = st.sidebar.number_input('Pregnancies', 0, 20, 1)
    glucose = st.sidebar.number_input('Glucose', 0, 200, 100)
    bp = st.sidebar.number_input('Blood Pressure', 0, 150, 70)
    skin = st.sidebar.number_input('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.number_input('Insulin', 0, 900, 80)
    bmi = st.sidebar.number_input('BMI', 0.0, 70.0, 25.0)
    dpf = st.sidebar.number_input('Diabetes Pedigree Function', 0.0, 3.0, 0.5)
    age = st.sidebar.number_input('Age', 10, 100, 30)

    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skin,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    return pd.DataFrame([data])

df = user_input()

st.subheader('Input Data')
st.write(df)

# Make prediction
pred = model.predict(df)[0]
proba = model.predict_proba(df)[0]

st.subheader('Prediction Result')
st.write('🧬 Diabetic' if pred == 1 else '✅ Not Diabetic')

st.subheader('Probability')
st.write(f"Not Diabetic: {proba[0]:.2f}")
st.write(f"Diabetic: {proba[1]:.2f}")

# Coefficients instead of feature_importances_
st.subheader('Feature Importance (Model Coefficients)')
coefficients = model.coef_[0]
features = df.columns

fig, ax = plt.subplots()
ax.barh(features, coefficients)
ax.set_xlabel("Coefficient Value")
st.pyplot(fig)
