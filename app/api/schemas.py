"""Pydantic schemas for the prediction API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Feature values keyed by name (e.g. lag_1, total_price, price_diff)",
    )


class PredictResponse(BaseModel):
    prediction: float
    model_version: str
    model_type: str
