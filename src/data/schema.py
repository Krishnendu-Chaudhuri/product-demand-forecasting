"""Pandera schemas for raw data validation."""

from __future__ import annotations

import pandera as pa
from pandera.typing import Series


class RawDataSchema(pa.DataFrameModel):
    """Schema for raw CSV columns before date parsing."""

    record_ID: Series[int] = pa.Field(ge=1)
    week: Series[str]
    store_id: Series[int]
    sku_id: Series[int]
    total_price: Series[float] = pa.Field(nullable=True)
    base_price: Series[float]
    is_featured_sku: Series[int] = pa.Field(isin=[0, 1])
    is_display_sku: Series[int] = pa.Field(isin=[0, 1])
    units_sold: Series[int] = pa.Field(ge=0)

    class Config:
        coerce = True
        strict = False


RAW_SCHEMA = RawDataSchema.to_schema()
