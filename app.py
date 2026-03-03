import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Anemia Prediction", page_icon="🩸")

st.title("🩸 Anemia Prediction System")

# Load model
data = joblib.load("anemia_model.pkl")
model = data["model"]
scaler = data["scaler"]

# Load dataset to get feature names
df = pd.read_csv("anemia.csv")
feature_columns = df.drop("Result", axis=1).columns

st.subheader("Enter Patient Details")

user_inputs = []

for col in feature_columns:
    value = st.number_input(f"{col}", value=0.0)
    user_inputs.append(value)

if st.button("Predict"):

    input_array = np.array([user_inputs])
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error("⚠️ Patient HAS Anemia")
    else:
        st.success("✅ Patient DOES NOT Have Anemia")

    st.write(f"Probability: {round(probability*100,2)}%")