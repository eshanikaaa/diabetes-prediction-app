import streamlit as st
import joblib
import numpy as np

st.title("🩺 Diabetes Prediction App")

# Load the trained model
import os
model_path = os.path.join(os.path.dirname(__file__), "diabetes_model.pkl")
model = joblib.load(model_path)

# Input form
st.header("Enter Patient Data")
pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose", 0, 200)
blood_pressure = st.number_input("Blood Pressure", 0, 150)
skin_thickness = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 1, 120)

if st.button("Predict"):
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                          insulin, bmi, dpf, age]])
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("🚨 The person is likely to have diabetes.")
    else:
        st.success("✅ The person is not likely to have diabetes.")

