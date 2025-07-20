import streamlit as st
import joblib
import os

st.write("✅ App start...")

# Check file existence
st.write("Looking for model file in current folder:")
st.write(os.listdir('.'))

# Try loading the model
try:
    model = joblib.load("diabetes_model.pkl")
    st.write("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

st.write("✅ Past model load — Reach here.")

# If all good, show a basic prediction test
st.write("Running a test prediction...")
try:
    pred = model.predict([[0,0,0,0,0,0.0,0.0,0]])
    st.write("✅ Test prediction:", pred)
except Exception as e:
    st.error(f"❌ Error predicting: {e}")


