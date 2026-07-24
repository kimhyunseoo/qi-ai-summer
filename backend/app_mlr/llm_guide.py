"""LLM-generated Power Usage Guide.

Falls back to a template-based guide automatically if OPENAI_API_KEY is
missing, invalid, expired, or the call fails for any reason (rate limit,
credit exhausted, network error) -- the forecast API must never break just
because the LLM call failed.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

_API_KEY = os.environ.get("OPENAI_API_KEY")
_client = None
if _API_KEY:
    from openai import OpenAI

    _client = OpenAI(api_key=_API_KEY)


def _peak_window(peak_h: int, peak_m: int) -> tuple[str, str]:
    """Peak time +/- 1 hour, clamped to the 05:00-19:30 forecast window."""
    start_h = max(5, peak_h - 1)
    end_h = min(19, peak_h + 1)
    return f"{start_h:02d}:{peak_m:02d}", f"{end_h:02d}:{peak_m:02d}"


def _template_guide(peak_h: int, peak_m: int, low_h: int, low_m: int) -> str:
    start, end = _peak_window(peak_h, peak_m)
    return (
        f"- Peak generation window {start}-{end} -- best window for washing machines, dryers, or EV charging\n"
        f"- Lowest generation around {low_h:02d}:{low_m:02d} -- avoid heavy appliance use here\n"
        "- Shift energy-intensive tasks earlier in the day when possible"
    )


def generate_usage_guide(
    target_date: str,
    total_kwh: float,
    vs_avg_pct: float,
    peak_h: int,
    peak_m: int,
    peak_kwh: float,
    low_h: int,
    low_m: int,
) -> str:
    if _client is None:
        return _template_guide(peak_h, peak_m, low_h, low_m)

    peak_start, peak_end = _peak_window(peak_h, peak_m)
    prompt = (
        f"Here is the solar generation forecast for {target_date}:\n"
        f"- Total generation: {total_kwh}kWh ({vs_avg_pct:+.1f}% vs yesterday)\n"
        f"- Peak window: {peak_start}-{peak_end} (peak ~{peak_kwh:.1f}kWh around {peak_h:02d}:{peak_m:02d})\n"
        f"- Lowest time: {low_h:02d}:{low_m:02d}\n\n"
        "Based on this, write a short power usage guide in English for a residential "
        "solar user as 3-4 concise bullet points (each starting with \"- \"). No intro "
        "or closing sentence, no headers -- just the bullets. Be specific and direct "
        "(exact times, exact recommendation), not filler. Cover: when to run high-load "
        "appliances (washing machine, dryer, EV charging), and when to avoid them."
    )

    try:
        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=180,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        # API key expired/exhausted/rate-limited/network error -> degrade gracefully
        return _template_guide(peak_h, peak_m, low_h, low_m)
