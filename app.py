import streamlit as st
import joblib
import numpy as np
import base64
import pandas as pd

df = pd.read_csv("dataset/heart.csv")

st.set_page_config(
    page_title = "Healthcare AI Suite",
    page_icon = "🏥",
    layout = "wide"
)
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

#custom css

st.markdown("""
<style>
.main {
    background-color: #e47b1a;
}
h1,h2,h3{
    color: white;
}

.stButton>button{
    width: 100%;
    height: 3.2em;
    border-radius:12px;
    border:none;
    background-color: #b74646;
    color:white;
    font-size:20px;
    font-weight:bold;
}

.stButton>button:hover{
    background-clor:#ff2b2b;
    color: white;
}
.block-container{
    padding-top:2rem;
}

</style>
""", unsafe_allow_html= True)


#st.title("AI Health Risk Prediction")

#age = st.number_input("Age", min_value = 0, max_value = 120, value = 30)
#bp = st.number_input("Blood Pressure", min_value = 0, max_value = 200, value = 120)
#chol = st.number_input("cholestrol", min_value = 0, max_value = 500, value = 200)
#st.sidebar.title("patient Details")
st.sidebar.title("🏥 Healthcare AI Suite")

page = st.sidebar.selectbox(
    "Navigate to:",
    ["AI Health Risk Prediction"]
)

st.sidebar.info("💡 Select a section from above to get started")

st.markdown(
    "<h1 style = 'color:White;'>AI Health Risk Prediction</h1>",unsafe_allow_html=True
)

st.subheader("Predict patient risk based on health parameters")
st.divider()

#input fields

tab1, tab2 = st.tabs(["Prediction", "Analytics"])

with tab1:
    col1,col2,col3 = st.columns(3)
    with col1:
        age = st.number_input("Age1")
        sex = st.selectbox("Sex", [0,1])
        cp = st.number_input("Chest Pain Type")

        trestbps = st.number_input("Resting Blood Pressure")
    with col2:
        chol = st.number_input("Cholesterol")

        fbs = st.selectbox("Fasting Blood Sugar", [0,1])

        restecg = st.number_input("Rest ECG")
    
        thalach = st.number_input("Max Heart Rate")
    with col3:
        exang = st.selectbox("Exercise Angina", [0,1])

        oldpeak = st.number_input("Oldpeak")

        slope = st.number_input("Slope")

        ca = st.number_input("CA")

        thal = st.number_input("Thal")
    st.divider()

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
        data = scaler.transform(data)
        with st.spinner("Analyzing patient data..."):
            prediction = model.predict(data)
        st.divider()
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
        
with tab2:
    st.header("Dataset Analytics")
    with st.expander("See Patient Details"):

        st.write(f"Age: {age}")
        st.write(f"Blood Pressure: {trestbps}")
        st.write(f"Cholesterol: {chol}")
        st.write(f"Heart Rate: {thalach}")

    st.line_chart(df["chol"])
    st.subheader("cholestrol Analysis")
    st.bar_chart(df["chol"])

    with st.expander("Dataset Info"):

        st.write(df.describe())