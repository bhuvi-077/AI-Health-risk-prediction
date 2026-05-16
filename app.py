import streamlit as st
import joblib
import numpy as np
import base64


def set_bg():
    with open("bg.jpg", "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    page_bg = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size:cover;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html = True)
set_bg()
model = joblib.load("models/health_model.pkl")
scaler = joblib.load("models/scaler.pkl")
st.title("AI Health Risk Prediction")

#age = st.number_input("Age", min_value = 0, max_value = 120, value = 30)
#bp = st.number_input("Blood Pressure", min_value = 0, max_value = 200, value = 120)
#chol = st.number_input("cholestrol", min_value = 0, max_value = 500, value = 200)

age = st.number_input("Age")

sex = st.selectbox("Sex", [0,1])

cp = st.number_input("Chest Pain Type")

trestbps = st.number_input("Resting Blood Pressure")

chol = st.number_input("Cholesterol")

fbs = st.selectbox("Fasting Blood Sugar", [0,1])

restecg = st.number_input("Rest ECG")

thalach = st.number_input("Max Heart Rate")

exang = st.selectbox("Exercise Angina", [0,1])

oldpeak = st.number_input("Oldpeak")

slope = st.number_input("Slope")

ca = st.number_input("CA")

thal = st.number_input("Thal")
data = [[
    age,
    sex,
    cp,
    trestbps,
    chol,
    fbs,
    restecg,
    thalach,
    exang,
    oldpeak,
    slope,
    ca,
    thal
]]
if st.button("predict"):
    #data = model.transform(data)
    prediction = model.predict(data)
    if prediction[0] == 0:
        st.error("High Health Risk")
    else:
        st.success("Low Health Risk")
