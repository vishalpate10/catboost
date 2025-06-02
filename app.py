import streamlit as st
import pickle
import numpy as np

# Load CatBoost model
with open('catboost.pkl', 'rb') as file:
    model = pickle.load(file)

# Page setup
st.set_page_config(page_title="Iris Flower Predictor", layout="centered")
st.title("🌸 Iris Flower Species Predictor")
st.write("Enter the flower measurements to predict its species:")

# Input fields for Iris features
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
sepal_width  = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
petal_width  = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

# Predict button
if st.button("Predict"):
    try:
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_data)[0]

        # Optional: show class name if it's encoded
        species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
        if isinstance(prediction, (int, np.integer)):
            prediction_text = species_map.get(prediction, f"Class {prediction}")
        else:
            prediction_text = prediction  # In case model returns string directly

        st.success(f"🌼 Predicted Species: **{prediction_text}**")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
