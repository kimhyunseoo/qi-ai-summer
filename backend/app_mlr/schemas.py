from pydantic import BaseModel


class SlotForecast(BaseModel):
    hour: int
    minute: int
    generation_kwh: float


class ForecastResponse(BaseModel):
    facility_name: str
    target_date: str  # YYYY-MM-DD
    model_name: str

    slots: list[SlotForecast]  # 05:00~19:30, 30-min steps (30 points)

    total_generation_kwh: float
    vs_avg_pct: float

    peak_hour: int
    peak_minute: int
    peak_generation_kwh: float

    lowest_hour: int
    lowest_minute: int
    lowest_generation_kwh: float

    usage_guide: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
