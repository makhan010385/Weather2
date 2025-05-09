import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib

st.title("🌾 5-Day Weather + Disease Forecast")

model = joblib.load("weather_multi_model.joblib")
disease_model = joblib.load("disease_classifier.joblib")
disease_le = joblib.load("disease_label_encoder.joblib")
location_encoder = joblib.load("weather_location_encoder.joblib")

location = st.text_input("Enter Location (case-sensitive)", "Jabalpur")
start_date = st.date_input("Select Start Date", datetime.today())
selected_variety = st.selectbox("Select Crop Variety", ["JS 95-60", "JS-335", "Shivalik", "JS93-05"])

if location not in location_encoder.classes_:
    st.error("❌ Location not found in trained model.")
else:
    location_code = location_encoder.transform([location])[0]
    forecast = []

    for i in range(1, 6):
        future_date = pd.to_datetime(start_date) + timedelta(days=i)
        day_of_year = future_date.timetuple().tm_yday
        year = future_date.year

        input_vector = np.array([[location_code, day_of_year, year]])
        prediction = model.predict(input_vector)[0]

        max_temp = round(prediction[0], 2)
        min_temp = round(prediction[1], 2)
        rainfall = round(prediction[2], 2)
        max_humidity = round(prediction[3], 2)
        min_humidity = round(prediction[4], 2)

        disease_input = {
            "Location_Code": location_code,
            "Max_Temp": max_temp,
            "Min_Temp": min_temp,
            "Rainfall": rainfall,
            "Max_Humidity": max_humidity,
            "Min_Humidity": min_humidity,
            "JS 95-60": 1 if selected_variety == "JS 95-60" else 0,
            "JS-335": 1 if selected_variety == "JS-335" else 0,
            "Shivalik": 1 if selected_variety == "Shivalik" else 0,
            "JS93-05": 1 if selected_variety == "JS93-05" else 0
        }

        disease_df = pd.DataFrame([disease_input])
        disease_pred = disease_model.predict(disease_df)[0]
        disease_name = disease_le.inverse_transform([disease_pred])[0]

        forecast.append({
            "Date": future_date.strftime("%Y-%m-%d"),
            "Max_Temp (°C)": max_temp,
            "Min_Temp (°C)": min_temp,
            "Rainfall (mm)": rainfall,
            "Max_Humidity (%)": max_humidity,
            "Min_Humidity (%)": min_humidity,
            "Predicted Disease": disease_name
        })

    st.subheader("📊 5-Day Forecast Table")
    forecast_df = pd.DataFrame(forecast)
    st.dataframe(forecast_df)
    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", data=csv, file_name="5_day_forecast.csv")
