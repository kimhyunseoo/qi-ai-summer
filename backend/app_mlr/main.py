from __future__ import annotations

import datetime as dt
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app_mlr.feature_engineering import TARGET_SLOTS, InsufficientHistoryError, build_feature_rows, load_history
from app_mlr.llm_guide import generate_usage_guide
from app_mlr.mock_model import predict
from app_mlr.schemas import ForecastResponse, HealthResponse, SlotForecast

MODEL_NAME = "Dacon-MLR-v1 (7-day lag / 30-min)"
FACILITY_NAME = "Dangjin Residential Plant"
LOCATION = "Dangjin, South Chungcheong"

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


@app.get("/api/forecast", response_model=ForecastResponse)
def forecast(date: str | None = Query(None, description="YYYY-MM-DD, demo/test용. 안 주면 내일 날짜")) -> ForecastResponse:
    target_date = dt.date.fromisoformat(date) if date else dt.date.today() + dt.timedelta(days=1)

    try:
        history = load_history(str(HISTORY_CSV_PATH))
        X = build_feature_rows(history, target_date)
    except InsufficientHistoryError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    values = predict(X)

    slots = [SlotForecast(hour=h, minute=m, generation_kwh=v) for (h, m), v in zip(TARGET_SLOTS, values)]

    total = round(sum(values) * 0.5, 1)  # 30-min slots -> kWh = sum(kW) * 0.5h
    avg_baseline_kwh = 430.8  # train 세트 일평균 총발전량(kWh) 실측 기준
    vs_avg_pct = round((total - avg_baseline_kwh) / avg_baseline_kwh * 100, 1)

    peak_idx = max(range(len(values)), key=lambda i: values[i])
    low_idx = min(range(len(values)), key=lambda i: values[i])
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
        location=LOCATION,
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
