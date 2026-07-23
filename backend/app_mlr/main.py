from __future__ import annotations

import datetime as dt
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app_mlr.feature_engineering import TARGET_SLOTS, InsufficientHistoryError, build_feature_rows, load_history
from app_mlr.llm_guide import generate_usage_guide
from app_mlr.mock_model import predict
from app_mlr.schemas import ForecastResponse, HealthResponse, SlotForecast

MODEL_NAME = "Dacon-RF-v1 (7-day lag / 30-min)"
FACILITY_NAME = "Residential Solar Plant"

HISTORY_CSV_PATH = Path(__file__).parent / "data" / "recent_data.csv"

app = FastAPI(title="Solarix Forecast API (MLR, mock)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before deploying
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=True, model_name=MODEL_NAME)


def _predict_day(history, target_date: dt.date) -> list[float]:
    X = build_feature_rows(history, target_date)
    return predict(X)


@app.get("/api/forecast", response_model=ForecastResponse)
def forecast(date: str | None = Query(None, description="YYYY-MM-DD, demo/test용. 안 주면 오늘 날짜")) -> ForecastResponse:
    target_date = dt.date.fromisoformat(date) if date else dt.date.today()
    history = load_history(str(HISTORY_CSV_PATH))

    try:
        values = _predict_day(history, target_date)
    except InsufficientHistoryError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    slots = [SlotForecast(hour=h, minute=m, generation_kwh=v) for (h, m), v in zip(TARGET_SLOTS, values)]

    total = round(sum(values) * 0.5, 1)  # 30-min slots -> kWh = sum(kW) * 0.5h

    # vs 전날 (같은 모델로 어제도 예측해서 비교; 어제치 history가 없으면 비교 생략)
    try:
        yesterday_values = _predict_day(history, target_date - dt.timedelta(days=1))
        yesterday_total = sum(yesterday_values) * 0.5
        vs_avg_pct = round((total - yesterday_total) / yesterday_total * 100, 1) if yesterday_total > 0 else 0.0
    except InsufficientHistoryError:
        vs_avg_pct = 0.0

    peak_idx = max(range(len(values)), key=lambda i: values[i])
    # 0보다 큰 슬롯 중 최소값 (없으면 그냥 전체 최소값으로 대체)
    positive_idxs = [i for i, v in enumerate(values) if v > 0]
    low_idx = min(positive_idxs, key=lambda i: values[i]) if positive_idxs else min(range(len(values)), key=lambda i: values[i])
    peak_h, peak_m = TARGET_SLOTS[peak_idx]
    low_h, low_m = TARGET_SLOTS[low_idx]

    usage_guide = generate_usage_guide(
        target_date=target_date.strftime("%Y-%m-%d"),
        total_kwh=total,
        vs_avg_pct=vs_avg_pct,
        peak_h=peak_h,
        peak_m=peak_m,
        peak_kwh=values[peak_idx],
        low_h=low_h,
        low_m=low_m,
    )

    return ForecastResponse(
        facility_name=FACILITY_NAME,
        target_date=target_date.strftime("%Y-%m-%d"),
        model_name=MODEL_NAME,
        slots=slots,
        total_generation_kwh=total,
        vs_avg_pct=vs_avg_pct,
        peak_hour=peak_h,
        peak_minute=peak_m,
        peak_generation_kwh=values[peak_idx],
        lowest_hour=low_h,
        lowest_minute=low_m,
        lowest_generation_kwh=values[low_idx],
        usage_guide=usage_guide,
    )
