import pickle
import numpy as np
import streamlit as st

st.set_page_config(page_title="Used Car Price Predictor", layout="centered")
st.title("Used Car Price Prediction System")
st.write("Craigslist Cars & Trucks ML Valuation Demo")

@st.cache_resource
def load_artifacts():
    with open("craigslist_car_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

model, encoders = load_artifacts()

year = st.number_input("Manufacturing Year", min_value=1995, max_value=2024, value=2020, step=1)
odometer = st.number_input("Odometer (Miles)", min_value=1, max_value=300000, value=45000, step=1000)
manufacturer = st.selectbox("Manufacturer", sorted(encoders["manufacturer"].classes_.tolist()))
condition = st.selectbox("Condition", sorted(encoders["condition"].classes_.tolist()))
fuel = st.selectbox("Fuel Type", sorted(encoders["fuel"].classes_.tolist()))
transmission = st.selectbox("Transmission", sorted(encoders["transmission"].classes_.tolist()))

if st.button("Predict Price"):
    input_data = np.array([[
        year,
        encoders["manufacturer"].transform([manufacturer])[0],
        encoders["condition"].transform([condition])[0],
        encoders["fuel"].transform([fuel])[0],
        odometer,
        encoders["transmission"].transform([transmission])[0],
    ]])
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Market Price: ${prediction:,.2f}")

#python -m streamlit run app.py