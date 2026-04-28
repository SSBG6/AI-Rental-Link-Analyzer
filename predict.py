import joblib
import pandas as pd

model = joblib.load("models/deal_model.pkl")

# your input
data = pd.DataFrame([{
    "area": "Berlin Mitte",
    "price": 600,
    "sqm": 45,
    "rooms": 2,
    "deposit": 1700,
    "furnished": 1,
    "bills_included": 1,
    "near_transport": 1,
    "viewing_available": 1,
}])

# 🔥 MUST recreate training features
data["price_per_sqm"] = data["price"] / data["sqm"]
data["deposit_months"] = data["deposit"] / data["price"]
data["room_size"] = data["sqm"] / data["rooms"]

# you also need area averages (from training data)
ref = pd.read_csv("data/listings.csv")
ref["price_per_sqm"] = ref["price"] / ref["sqm"]

area_avg = ref.groupby("area")["price_per_sqm"].mean()

data["area_avg_price_per_sqm"] = data["area"].map(area_avg)

data["discount_vs_area"] = (
    data["area_avg_price_per_sqm"] - data["price_per_sqm"]
) / data["area_avg_price_per_sqm"]

data["is_cheap_for_area"] = (data["discount_vs_area"] >= 0.15).astype(int)
data["is_deposit_reasonable"] = (data["deposit_months"] <= 3).astype(int)

data["value_score"] = (
    data["discount_vs_area"] * 100
    + data["bills_included"] * 5
    + data["near_transport"] * 5
    + data["viewing_available"] * 10
    - data["deposit_months"] * 5
)

# 🚀 prediction
prediction = model.predict(data)
probability = model.predict_proba(data)

print(prediction)
print(probability)