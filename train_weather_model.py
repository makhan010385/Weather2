import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("Model_CSV.csv")
df = df.dropna(subset=["Location", "Year", "SMW", "Max_Temp"])
df["DayOfYear"] = df["SMW"] * 7

le_location = LabelEncoder()
df["Location_Code"] = le_location.fit_transform(df["Location"])

features = ["Location_Code", "DayOfYear", "Year"]
targets = ["Max_Temp", "Min_Temp", "Rainfall", "Max_Humidity", "Min_Humidity"]

X = df[features]
Y = df[targets].apply(pd.to_numeric, errors="coerce").fillna(0)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, Y)

joblib.dump(model, "weather_multi_model.joblib")
joblib.dump(le_location, "weather_location_encoder.joblib")
print("✅ Weather model saved.")
