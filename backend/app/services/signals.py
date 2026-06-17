from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..config import settings

POSITIVE_TYPES = {"click", "cart_add", "purchase", "search", "view", "email_open", "push_click", "site_search"}
NEGATIVE_TYPES = {"ignore", "skip", "unsubscribe", "email_ignore", "ad_skip", "ad_ignore"}


def exponential_decay(age_hours: float, half_life: float | None = None) -> float:
    half_life = half_life or settings.decay_half_life_hours
    return 0.5 ** (age_hours / half_life)


def event_age_hours(event: dict[str, Any], now: datetime | None = None) -> float:
    now = now or datetime.now(timezone.utc)
    created = datetime.fromisoformat(event["created_at"])
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return max(0.0, (now - created).total_seconds() / 3600)


def event_weight(event: dict[str, Any], now: datetime | None = None) -> float:
    decay = exponential_decay(event_age_hours(event, now))
    event_type = event.get("type", "")
    sentiment = event.get("sentiment", "")

    if sentiment == "positive" or event_type in POSITIVE_TYPES:
        base = 1.0
        if event_type == "purchase":
            base = 3.0
        elif event_type in ("cart_add", "click"):
            base = 1.5
        return base * decay

    if sentiment == "negative" or event_type in NEGATIVE_TYPES:
        base = -1.2
        if event_type == "unsubscribe":
            base = -3.0
        return base * decay

    return 0.3 * decay


def weighted_signal_sum(events: list[dict[str, Any]], window_hours: float | None = None) -> float:
    now = datetime.now(timezone.utc)
    total = 0.0
    for event in events:
        age = event_age_hours(event, now)
        if window_hours is not None and age > window_hours:
            continue
        total += event_weight(event, now)
    return total


def build_time_decayed_features(
    customer_id: str,
    events: list[dict[str, Any]],
    product_id: str | None = None,
) -> dict[str, float]:
    customer_events = [e for e in events if e["customer_id"] == customer_id]
    if product_id:
        product_events = [e for e in customer_events if e["product_id"] == product_id]
    else:
        product_events = customer_events

    def count_type(types: set[str], hours: float) -> float:
        return sum(
            event_weight(e)
            for e in customer_events
            if e["type"] in types and event_age_hours(e) <= hours
        )

    def product_signal(types: set[str], hours: float) -> float:
        return sum(
            event_weight(e)
            for e in product_events
            if e["type"] in types and event_age_hours(e) <= hours
        )

    return {
        "clicks_24h": count_type({"click", "push_click"}, 24),
        "clicks_7d": count_type({"click", "push_click"}, 168),
        "cart_adds_7d": count_type({"cart_add"}, 168),
        "purchases_30d": count_type({"purchase"}, 720),
        "email_opens_30d": count_type({"email_open"}, 720),
        "searches_7d": count_type({"search", "site_search"}, 168),
        "views_7d": count_type({"view"}, 168),
        "negative_7d": abs(sum(
            event_weight(e)
            for e in customer_events
            if e.get("sentiment") == "negative" and event_age_hours(e) <= 168
        )),
        "product_clicks_7d": product_signal({"click"}, 168),
        "product_views_7d": product_signal({"view"}, 168),
        "product_cart_7d": product_signal({"cart_add"}, 168),
        "total_signal": weighted_signal_sum(customer_events),
    }
