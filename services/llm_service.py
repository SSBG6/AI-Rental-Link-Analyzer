import json
import requests

from app.config import LMSTUDIO_URL, MODEL_NAME


def extract_listing_details(raw_text: str):
    prompt = f"""
Extract apartment listing details and return ONLY valid JSON.

Required JSON format:
{{
  "area": "",
  "price": 0,
  "sqm": 0,
  "rooms": 0,
  "deposit": 0,
  "furnished": 0,
  "bills_included": 0,
  "near_transport": 0,
  "viewing_available": 0
}}

Rules:
- Return JSON only.
- No explanation.
- No markdown.
- Numbers only for price, sqm, rooms, deposit.
- Yes = 1, No = 0.
- If unknown, use 0.
- Area should be like "Berlin Mitte", "Berlin Kreuzberg", etc.

TEXT:
{raw_text}
"""

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 250
    }

    response = requests.post(LMSTUDIO_URL, json=payload)
    response.raise_for_status()

    result = response.json()["choices"][0]["message"]["content"]

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        start = result.find("{")
        end = result.rfind("}") + 1

        if start == -1 or end == 0:
            return None

        return json.loads(result[start:end])


def generate_final_explanation(listing, prediction_label, confidence, best_avg, best_deals_count):
    if best_avg is None:
        prompt = f"""
You are a rental deal analyst.

Write a short, natural verdict for the user.

Current listing:
Area: {listing["area"]}
Price: €{listing["price"]}
Size: {listing["sqm"]} sqm
Rooms: {listing["rooms"]}
€/sqm: €{round(listing["price_per_sqm"], 2)}
Value score: {round(listing["value_score"], 2)}

Deal result:
Verdict: {prediction_label}
Confidence: {confidence}%

Instructions:
- Do NOT mention missing data.
- Do NOT mention ChromaDB, database, benchmark, or lack of comparison.
- Explain whether this seems like a good, average, or bad deal.
- Keep it short, confident, and useful.
"""

    else:
        prompt = f"""
You are a rental deal analyst.

Write a clear, concise verdict.

Current listing:
Area: {listing["area"]}
Price: €{listing["price"]}
Size: {listing["sqm"]} sqm
Rooms: {listing["rooms"]}
€/sqm: €{round(listing["price_per_sqm"], 2)}
Value score: {round(listing["value_score"], 2)}

Deal result:
Verdict: {prediction_label}
Confidence: {confidence}%

Best deal comparison:
Known good deals in same area: {best_deals_count}
Average €/sqm of known good deals: €{round(best_avg, 2)}

Instructions:
- Explain whether this is a good deal.
- Say what a good deal for a similar place would cost.
- Keep it short and useful.
"""

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 500
    }

    response = requests.post(LMSTUDIO_URL, json=payload)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]