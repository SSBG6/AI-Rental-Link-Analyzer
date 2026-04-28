import pandas as pd
import json

CSV_PATH = "exports/deals.csv"
JSON_PATH = "exports/deals.json"


def convert_csv_to_json():
    df = pd.read_csv(CSV_PATH)

    deals = df.to_dict(orient="records")

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(deals, f, indent=4, ensure_ascii=False)

    print(f"Converted {len(deals)} deals to {JSON_PATH}")


if __name__ == "__main__":
    convert_csv_to_json()