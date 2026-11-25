# pyright: reportMissingModuleSource=false

import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("vehicles.csv")

cols = ["price", "year", "manufacturer", "condition", "fuel", "odometer", "transmission"]
df = df[cols].dropna()

df = df[(df["price"] >= 500) & (df["price"] <= 80000)]
df = df[(df["year"] >= 1995) & (df["year"] <= 2024)]
df = df[(df["odometer"] > 0) & (df["odometer"] <= 300000)]

cat_cols = ["manufacturer", "condition", "fuel", "transmission"]
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"R2 Score: {r2_score(y_test, preds):.4f}")
print(f"MAE: ${mean_absolute_error(y_test, preds):.2f}")

with open("craigslist_car_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)