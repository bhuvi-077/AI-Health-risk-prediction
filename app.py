import streamlit as st
import joblib
import numpy as np
import base64
import pandas as pd

df = pd.read_csv("dataset/heart.csv")


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
st.markdown(
    "<h1 style = 'color:White;'>AI Health Risk Prediction</h1>",unsafe_allow_html=True
)
#st.title("AI Health Risk Prediction")

#age = st.number_input("Age", min_value = 0, max_value = 120, value = 30)
#bp = st.number_input("Blood Pressure", min_value = 0, max_value = 200, value = 120)
#chol = st.number_input("cholestrol", min_value = 0, max_value = 500, value = 200)
#st.sidebar.title("patient Details")
tab1, tab2 = st.tabs(["Prediction", "Analytics"])

with tab1:
    col1,col2 = st.columns(2)
    with col1:
        age = st.number_input("Age")
        sex = st.selectbox("Sex", [0,1])
        cp = st.number_input("Chest Pain Type")

        trestbps = st.number_input("Resting Blood Pressure")

        chol = st.number_input("Cholesterol")

        fbs = st.selectbox("Fasting Blood Sugar", [0,1])

        restecg = st.number_input("Rest ECG")
    with col2:
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
        #st.balloons()
        with st.spinner("Analyzing patient data..."):
            prediction = model.predict(data)
        col1,col2 = st.columns(2)
        with col1:
            st.metric("Age", age)
            st.metric("Blood pressure", trestbps)
            st.metric("Cholesterol", chol)
        if prediction[0] == 0:
            st.error("High Health Risk")
            with col2:
                st.image("high risk.jpg", width = 400)
        else:
            st.success("Good Health")
            with col2:
                st.image("good health.jpg", width = 400)
            st.balloons()
        st.subheader("cholestrol Analysis")
        st.bar_chart(df["chol"])
with tab2:
    st.header("Dataset Analytics")

    st.line_chart(df["chol"])

    with st.expander("Dataset Info"):

        st.write(df.describe())