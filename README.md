# 🩺 Diabetes Prediction App

A simple web application built with **Streamlit** that predicts whether a person is likely to have diabetes based on medical inputs. It uses a trained **Logistic Regression model** on the PIMA Indians Diabetes dataset.

---

## Features

- User-friendly web interface
- Input fields for health indicators (e.g., glucose, insulin, BMI)
- Predicts diabetes risk using machine learning
- Deployed online using **Streamlit Cloud**

---

## How to Use

1. Visit the live app 👉 [Click here](https://diabetes-prediction-app-jjbxpthkppqvw7aqj32rum.streamlit.app/)
2. Enter values for:
   - Pregnancies
   - Glucose
   - Blood Pressure
   - Skin Thickness
   - Insulin
   - BMI
   - Diabetes Pedigree Function
   - Age
3. Click **Predict**
4. Get instant results!

---

## Model Info

- **Algorithm:** Logistic Regression  
- **Dataset:** [PIMA Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
- **Training Script:** `train_model.py`

---

## 🛠️ Requirements

See [`requirements.txt`](requirements.txt)

Install locally:
```bash
pip install -r requirements.txt
