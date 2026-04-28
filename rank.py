import os
import pandas as pd
import numpy as np

DATA_PATH = "data/listings.csv"
EXPORT_PATH = "exports/top_3_deals_by_area_price_category.csv"


def clean_data(df):
    df = df.copy()

    df = df.drop_duplicates()

    required_columns = [
        "area",
        "price",
        "sqm",
        "rooms",
        "deposit",
        "furnished",
        "bills_included",
        "near_transport",
        "viewing_available",
    ]

    df = df.dropna(subset=required_columns)

    df = df[
        (df["price"] > 0) &
        (df["sqm"] > 0) &
        (df["rooms"] > 0) &
        (df["deposit"] >= 0)
    ]

    return df


def add_deal_features(df):
    df = df.copy()

    df["price_per_sqm"] = df["price"] / df["sqm"]
    df["deposit_months"] = df["deposit"] / df["price"]
    df["room_size"] = df["sqm"] / df["rooms"]

    df["area_avg_price_per_sqm"] = (
        df.groupby("area")["price_per_sqm"].transform("mean")
    )

    df["discount_vs_area"] = (
        df["area_avg_price_per_sqm"] - df["price_per_sqm"]
    ) / df["area_avg_price_per_sqm"]

    df["deal_score"] = (
        df["discount_vs_area"] * 100
        + df["bills_included"] * 8
        + df["near_transport"] * 8
        + df["viewing_available"] * 10
        + df["furnished"] * 4
        - df["deposit_months"] * 6
    )

    return df


def remove_bad_deals(df):
    df = df.copy()

    df = df[
        (df["price_per_sqm"] <= df["area_avg_price_per_sqm"] * 1.15) &
        (df["deposit_months"] <= 3) &
        (df["deal_score"] > 0)
    ]

    return df


def add_price_category(df):
    df = df.copy()

    def categorize(group):
        group = group.copy()

        if len(group) < 3:
            group["price_category"] = "medium"
            return group

        group["price_category"] = pd.qcut(
            group["price"],
            q=3,
            labels=["budget", "medium", "premium"],
            duplicates="drop"
        )

        return group

    return df.groupby("area", group_keys=False).apply(categorize)


def export_top_3_deals(df):
    df = df.copy()

    top_deals = (
        df.sort_values(
            by=["area", "price_category", "deal_score"],
            ascending=[True, True, False]
        )
        .groupby(["area", "price_category"])
        .head(3)
    )

    export_columns = [
        "area",
        "price_category",
        "price",
        "sqm",
        "rooms",
        "deposit",
        "price_per_sqm",
        "area_avg_price_per_sqm",
        "discount_vs_area",
        "deposit_months",
        "room_size",
        "furnished",
        "bills_included",
        "near_transport",
        "viewing_available",
        "deal_score",
    ]

    if "url" in top_deals.columns:
        export_columns.append("url")

    os.makedirs("exports", exist_ok=True)
    top_deals[export_columns].to_csv(EXPORT_PATH, index=False)

    print(f"\nExported top deals to: {EXPORT_PATH}")
    print(top_deals[["area", "price_category", "price", "sqm", "deal_score"]])


def main():
    df = pd.read_csv(DATA_PATH)

    print("Loaded dataset:", df.shape)

    df = clean_data(df)
    print("After cleaning:", df.shape)

    df = add_deal_features(df)

    df = remove_bad_deals(df)
    print("After removing bad deals:", df.shape)

    df = add_price_category(df)

    export_top_3_deals(df)


if __name__ == "__main__":
    main()