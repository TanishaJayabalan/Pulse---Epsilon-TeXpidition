from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

FatigueCause = str  # channel_fatigue | timing_mismatch | offer_fatigue | none


def detect_fatigue_signals(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Heuristic fatigue detection per spec thresholds."""
    sorted_events = sorted(events, key=lambda e: e["created_at"], reverse=True)

    channel_consecutive: dict[str, int] = defaultdict(int)
    offer_ignores: dict[str, int] = defaultdict(int)
    morning_open_days: set[str] = set()
    morning_ignore_days: set[str] = set()

    email_streak = 0
    for event in sorted_events:
        channel = event["channel"]
        event_type = event["type"]
        created = datetime.fromisoformat(event["created_at"])
        day_key = created.date().isoformat()
        hour = event.get("hour_of_day", created.hour)

        is_negative = event.get("sentiment") == "negative" or event_type in {
            "ignore", "email_ignore", "ad_skip", "unsubscribe"
        }

        if channel == "email" and is_negative:
            email_streak += 1
        elif channel == "email" and not is_negative:
            break

        if is_negative and event_type in ("ignore", "email_ignore"):
            offer_ignores[event.get("product_id", "unknown")] += 1

        if 9 <= hour <= 11:
            if event_type == "email_open":
                morning_open_days.add(day_key)
            elif is_negative:
                morning_ignore_days.add(day_key)

        if is_negative:
            channel_consecutive[channel] += 1

    channel_fatigue = email_streak >= 5
    offer_fatigue = max(offer_ignores.values(), default=0) >= 3
    timing_mismatch = len(morning_ignore_days) >= 5 and len(morning_open_days) == 0

    causes: list[FatigueCause] = []
    if channel_fatigue:
        causes.append("channel_fatigue")
    if timing_mismatch:
        causes.append("timing_mismatch")
    if offer_fatigue:
        causes.append("offer_fatigue")

    primary_cause: FatigueCause = causes[0] if causes else "none"

    fatigue_score = min(1.0, (
        (email_streak / 5) * 0.4
        + (max(offer_ignores.values(), default=0) / 3) * 0.35
        + (len(morning_ignore_days) / 5) * 0.25
    ))

    return {
        "fatigue_score": round(fatigue_score, 3),
        "primary_cause": primary_cause,
        "causes": causes or ["none"],
        "email_ignore_streak": email_streak,
        "max_offer_ignores": max(offer_ignores.values(), default=0),
        "morning_ignore_days": len(morning_ignore_days),
        "channel_consecutive": dict(channel_consecutive),
        "channel_fatigue": channel_fatigue,
        "offer_fatigue": offer_fatigue,
        "timing_mismatch": timing_mismatch,
    }
