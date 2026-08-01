"""Pydantic schemas for request/response validation."""
from pydantic import BaseModel
from typing import Optional, List


class PredictRequest(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float
    daily_return: Optional[float] = 0.0
    price_range: Optional[float] = None
    ma_7: Optional[float] = None
    ma_30: Optional[float] = None
    ma_90: Optional[float] = None
    volatility_7d: Optional[float] = None
    month: Optional[int] = None
    day_of_week: Optional[int] = None
    quarter: Optional[int] = None


class PredictResponse(BaseModel):
    predicted_price: float
    rf_prediction: float
    xgb_prediction: float
    confidence_lower: float
    confidence_upper: float


class ForecastPoint(BaseModel):
    date: str
    predicted: float
    lower_95: float
    upper_95: float
    horizon_days: int


class MetricsResponse(BaseModel):
    training_metrics: dict
    test_metrics: dict
    full_metrics: dict


class HistoryPoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    ma_7: Optional[float]
    ma_30: Optional[float]
    ma_90: Optional[float]
