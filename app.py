import streamlit as st
import joblib
import numpy as np

model = joblib.load("health_model.pkl")

st.title("AI Health Risk Prediction")

age = st.number_input("Age", min_value = 0, max_value = 120, value = 30)
bp = st.number_input("Blood Pressure", min_value = 0, max_value = 200, value = 120)
chol = st.number_input("cholestrol", min_value = 0, max_value = 500, value = 200)

if st.button("predict"):
    data = np.array([[age, bp, chol]])
    prediction = model.predict(data)
    if prediction[0] == 1:
        st.error("High Health Risk")
    else:
        st.success("Low Health Risk")
