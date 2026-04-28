from fastapi import APIRouter, HTTPException

from app.models.schemas import DealRequest
from app.services.scraper_service import scrape_listing
from app.services.llm_service import extract_listing_details, generate_final_explanation
from app.services.prediction_service import predict_deal
from app.services.chroma_service import (
    get_best_deals_same_area,
    get_best_deals_avg_price_per_sqm
)


router = APIRouter(
    prefix="/deals",
    tags=["Deals"]
)


@router.post("/analyze")
def analyze_deal(request: DealRequest):
    try:
        raw_text = scrape_listing(request.url)

        details = extract_listing_details(raw_text)

        if details is None:
            raise HTTPException(
                status_code=400,
                detail="Could not extract listing details from the URL."
            )

        prediction_result = predict_deal(details)

        listing = prediction_result["listing"]
        area = listing["area"]

        best_deals = get_best_deals_same_area(area)
        best_avg = get_best_deals_avg_price_per_sqm(best_deals)

        final_explanation = generate_final_explanation(
            listing=listing,
            prediction_label=prediction_result["prediction_label"],
            confidence=prediction_result["confidence"],
            best_avg=best_avg,
            best_deals_count=len(best_deals)
        )

        return {
            "input_url": request.url,
            "extracted_details": details,
            "prediction": prediction_result["prediction_label"],
            "confidence": prediction_result["confidence"],
            "probabilities": prediction_result["probabilities"],
            "price_per_sqm": round(listing["price_per_sqm"], 2),
            "value_score": round(listing["value_score"], 2),
            "best_deals_found": len(best_deals),
            "best_deals_avg_price_per_sqm": round(best_avg, 2) if best_avg else None,
            "final_explanation": final_explanation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))