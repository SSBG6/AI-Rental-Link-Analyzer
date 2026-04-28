import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


DATA_PATH = "data/listings.csv"
MODEL_PATH = "models/deal_model.pkl"


def train_model():
    df = pd.read_csv(DATA_PATH)

    print("\nDataset loaded")
    print(df.head())
    print("\nDataset shape:", df.shape)

    # New pattern features
    df["price_per_sqm"] = df["price"] / df["sqm"]
    df["deposit_months"] = df["deposit"] / df["price"]
    df["room_size"] = df["sqm"] / df["rooms"]

    df["area_avg_price_per_sqm"] = df.groupby("area")["price_per_sqm"].transform("mean")

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

    X = df[
        [
            "area",
            "price",
            "sqm",
            "rooms",
            "deposit",
            "furnished",
            "bills_included",
            "near_transport",
            "viewing_available",

            "price_per_sqm",
            "deposit_months",
            "room_size",
            "area_avg_price_per_sqm",
            "discount_vs_area",
            "is_cheap_for_area",
            "is_deposit_reasonable",
            "value_score",
        ]
    ]

    y = df["label"]

    print("\nLabel balance:")
    print(y.value_counts())

    categorical_features = ["area"]

    numeric_features = [
        "price",
        "sqm",
        "rooms",
        "deposit",
        "furnished",
        "bills_included",
        "near_transport",
        "viewing_available",

        "price_per_sqm",
        "deposit_months",
        "room_size",
        "area_avg_price_per_sqm",
        "discount_vs_area",
        "is_cheap_for_area",
        "is_deposit_reasonable",
        "value_score",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("area_encoder", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numbers", "passthrough", numeric_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        class_weight="balanced"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=42,
        stratify=y
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    print("\nModel trained.")

    print("\nAccuracy:")
    print(accuracy_score(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, predictions)
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    print("\nCross Validation Accuracy:")
    scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(scores)
    print("Mean CV Accuracy:", scores.mean())

    print("\nExample prediction test:")

    sample_listing = pd.DataFrame([
        {
            "area": "Berlin Mitte",
            "price": 850,
            "sqm": 45,
            "rooms": 2,
            "deposit": 1700,
            "furnished": 1,
            "bills_included": 1,
            "near_transport": 1,
            "viewing_available": 1,
        }
    ])

    # Add same pattern features to sample listing
    sample_listing["price_per_sqm"] = sample_listing["price"] / sample_listing["sqm"]
    sample_listing["deposit_months"] = sample_listing["deposit"] / sample_listing["price"]
    sample_listing["room_size"] = sample_listing["sqm"] / sample_listing["rooms"]

    area_avg_map = df.groupby("area")["price_per_sqm"].mean()
    sample_listing["area_avg_price_per_sqm"] = sample_listing["area"].map(area_avg_map)

    sample_listing["discount_vs_area"] = (
        sample_listing["area_avg_price_per_sqm"] - sample_listing["price_per_sqm"]
    ) / sample_listing["area_avg_price_per_sqm"]

    sample_listing["is_cheap_for_area"] = (
        sample_listing["discount_vs_area"] >= 0.15
    ).astype(int)

    sample_listing["is_deposit_reasonable"] = (
        sample_listing["deposit_months"] <= 3
    ).astype(int)

    sample_listing["value_score"] = (
        sample_listing["discount_vs_area"] * 100
        + sample_listing["bills_included"] * 5
        + sample_listing["near_transport"] * 5
        + sample_listing["viewing_available"] * 10
        - sample_listing["deposit_months"] * 5
    )

    sample_prediction = pipeline.predict(sample_listing)
    sample_probability = pipeline.predict_proba(sample_listing)

    print("Prediction:", sample_prediction[0])
    print("Prediction probabilities:", sample_probability)

    print("\nFeature Importance:")

    trained_preprocessor = pipeline.named_steps["preprocessor"]
    trained_model = pipeline.named_steps["model"]

    area_names = trained_preprocessor.named_transformers_["area_encoder"].get_feature_names_out(["area"])

    feature_names = list(area_names) + numeric_features

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": trained_model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print(importance_df)

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()