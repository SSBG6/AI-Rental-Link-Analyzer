import pandas as pd
import joblib

from app.config import DATA_PATH, MODEL_PATH


CLASSES = {
    0: "BAD DEAL",
    1: "AVERAGE DEAL",
    2: "GOOD DEAL"
}


def prepare_features(df, reference_df):
    df["price_per_sqm"] = df["price"] / df["sqm"]
    df["deposit_months"] = df["deposit"] / df["price"]
    df["room_size"] = df["sqm"] / df["rooms"]

    reference_df["price_per_sqm"] = reference_df["price"] / reference_df["sqm"]
    area_avg = reference_df.groupby("area")["price_per_sqm"].mean()

    df["area_avg_price_per_sqm"] = df["area"].map(area_avg)

    if df["area_avg_price_per_sqm"].isna().any():
        df["area_avg_price_per_sqm"] = reference_df["price_per_sqm"].mean()

    df["discount_vs_area"] = (
        df["area_avg_price_per_sqm"] - df["price_per_sqm"]
    ) / df["area_avg_price_per_sqm"]

    df["is_cheap_for_area"] = (df["discount_vs_area"] >= 0.15).astype(int)
    df["is_deposit_reasonable"] = (df["deposit_months"] <= 3).astype(int)

    df["value_score"] = (
        df["discount_vs_area"] * 100
        + df["bills_included"] * 5
        + df["near_transport"] * 5
        + df["viewing_available"] * 10
        - df["deposit_months"] * 5
    )

    return df


def predict_deal(details: dict):
    model = joblib.load(MODEL_PATH)
    reference_df = pd.read_csv(DATA_PATH)

    listing_df = pd.DataFrame([details])
    listing_df = prepare_features(listing_df, reference_df)

    prediction = model.predict(listing_df)[0]
    probabilities = model.predict_proba(listing_df)[0]

    prediction_label = CLASSES.get(prediction, str(prediction))
    confidence = round(probabilities[prediction] * 100, 2)

    return {
        "listing_df": listing_df,
        "listing": listing_df.iloc[0].to_dict(),
        "prediction": prediction,
        "prediction_label": prediction_label,
        "confidence": confidence,
        "probabilities": probabilities.tolist()
    }