"""Streaming simulation (Section 5.2): processes the past_rates time-series
month by month, as if records arrived incrementally from a Kafka / Spark
Structured Streaming source. Reveals the seasonal revenue pattern shown in
reports/figures/figure13_streaming_trend.png.
"""
from __future__ import annotations

import time

import pandas as pd


def run_streaming_simulation(rates: pd.DataFrame) -> pd.DataFrame:
    rates = rates.copy()
    for col in ("revenue", "occupancy", "rate_avg"):
        if col in rates.columns:
            rates[col] = pd.to_numeric(rates[col], errors="coerce")
    rates["date"] = pd.to_datetime(rates["date"])
    rates = rates.sort_values("date")

    monthly = []
    t0 = time.time()
    for period, group in rates.groupby(rates["date"].dt.to_period("M")):
        monthly.append({
            "period": str(period),
            "records": len(group),
            "total_revenue": group["revenue"].sum(),
            "avg_occupancy": group["occupancy"].mean(),
            "avg_rate": group["rate_avg"].mean(),
            "unique_listings": group["listing_id"].nunique(),
        })
    elapsed = time.time() - t0
    result = pd.DataFrame(monthly)
    result.attrs["elapsed_seconds"] = elapsed
    return result
