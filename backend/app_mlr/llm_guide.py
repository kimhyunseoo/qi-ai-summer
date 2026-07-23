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


def _template_guide(peak_h: int, peak_m: int, low_h: int, low_m: int) -> str:
    return (
        f"Today's generation is expected to peak around {peak_h:02d}:{peak_m:02d}. "
        "High-load appliances such as washing machines, dryers, or electric vehicle charging "
        "are recommended during this window to make the most of surplus generation.\n\n"
        f"Generation is expected to be minimal around {low_h:02d}:{low_m:02d}. "
        "Please plan energy-intensive tasks earlier in the day."
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

    prompt = (
        f"Here is the solar generation forecast for {target_date}:\n"
        f"- Total generation: {total_kwh}kWh ({vs_avg_pct:+.1f}% vs yesterday)\n"
        f"- Peak time: {peak_h:02d}:{peak_m:02d} (about {peak_kwh:.1f}kWh)\n"
        f"- Lowest time: {low_h:02d}:{low_m:02d}\n\n"
        "Based on this, write a 2-paragraph power usage guide in English for a "
        "residential solar user, in a friendly and helpful tone. Give specific "
        "recommendations on when to run high-load appliances like washing machines, "
        "dryers, or EV charging."
    )

    try:
        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        # API key expired/exhausted/rate-limited/network error -> degrade gracefully
        return _template_guide(peak_h, peak_m, low_h, low_m)
