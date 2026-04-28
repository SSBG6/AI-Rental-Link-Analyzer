from pydantic import BaseModel


class DealRequest(BaseModel):
    url: str


class DealResponse(BaseModel):
    prediction: str
    confidence: float
    final_explanation: str