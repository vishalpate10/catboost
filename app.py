import streamlit as st
import pickle
import numpy as np

# Load the CatBoost model
with open('catboost.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("CatBoost Model Prediction")

f1 = st.number_input("Enter Feature 1", value=0.0)
f2 = st.number_input("Enter Feature 2", value=0.0)

if st.button("Predict"):
    features = np.array([[f1, f2]])
    prediction = model.predict(features)
    st.success(f"Prediction: {prediction[0]}")
