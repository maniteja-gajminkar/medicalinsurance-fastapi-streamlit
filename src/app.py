import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/insurance_model.pkl")

st.title("💡 Medical Insurance Cost Predictor")

# Input fields
age = st.slider("Age", 18, 65, 30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.slider("BMI", 15.0, 45.0, 25.0)
children = st.slider("Number of Children", 0, 5, 1)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Predict
input_df = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "children": children,
    "smoker": smoker,
    "region": region
}])

if st.button("Predict Insurance Cost"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Charge: ₹{round(prediction, 2)}")