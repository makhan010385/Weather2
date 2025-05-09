import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("Model_CSV.csv")
df.replace('-', pd.NA, inplace=True)
df = df.dropna(subset=["Location", "Disease", "Max_Temp", "Min_Temp", "Rainfall"])

numeric_cols = ["Max_Temp", "Min_Temp", "Rainfall", "Max_Humidity", "Min_Humidity"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=numeric_cols)

le_location = LabelEncoder()
df["Location_Code"] = le_location.fit_transform(df["Location"])

le_disease = LabelEncoder()
df["Disease_Label"] = le_disease.fit_transform(df["Disease"])

varieties = ["JS 95-60", "JS-335", "Shivalik", "JS93-05"]
for v in varieties:
    df[v] = df[v].fillna(0)

features = ["Location_Code"] + numeric_cols + varieties
X = df[features]
y = df["Disease_Label"]

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

joblib.dump(clf, "disease_classifier.joblib")
joblib.dump(le_location, "disease_location_encoder.joblib")
joblib.dump(le_disease, "disease_label_encoder.joblib")
print("✅ Disease classifier saved.")
