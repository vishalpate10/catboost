import streamlit as st
import pickle
import numpy as np

# Load trained CatBoost model
with open('catboost.pkl', 'rb') as file:
    model = pickle.load(file)

# Species label mapping
species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

# Set up Streamlit UI
st.set_page_config(page_title="Iris Flower Classifier 🌸", layout="centered")
st.title("🌼 Iris Flower Species Predictor")
st.write("Enter the flower's measurements below to predict its species:")

# Input fields for 4 Iris features
sepal_length = st.number_input("Sepal Length (cm)", value=5.1, format="%.2f")
sepal_width = st.number_input("Sepal Width (cm)", value=3.5, format="%.2f")
petal_length = st.number_input("Petal Length (cm)", value=1.4, format="%.2f")
petal_width = st.number_input("Petal Width (cm)", value=0.2, format="%.2f")

# Predict button
if st.button("Predict"):
    try:
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        pred_class = model.predict(features)[0]

        # Map prediction to flower name
        flower_name = species_map.get(int(pred_class), f"Unknown ({pred_class})")

        st.success(f"🌸 Predicted Flower: **{flower_name}**")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
