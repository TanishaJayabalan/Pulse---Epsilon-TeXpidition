from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class PeopleCloudMock:
    """Simulates Epsilon PeopleCloud identity + preference resolution."""

    def resolve(self, customer_id: str, cookie_id: str | None = None) -> dict[str, Any]:
        from ..store import store

        profile = store.get_customer(customer_id)
        if profile:
            return {
                "customer_id": profile["customer_id"],
                "source": "peoplecloud",
                "match_confidence": profile.get("match_confidence", 0.9),
                "profile": profile,
            }

        if cookie_id:
            return {
                "customer_id": f"anon_{cookie_id[:8]}",
                "source": "cookie_fallback",
                "match_confidence": 0.35,
                "profile": {
                    "customer_id": f"anon_{cookie_id[:8]}",
                    "interests": [],
                    "context": {"device": "unknown", "time_of_day": _time_of_day()},
                    "fatigue_score": 0.0,
                },
            }

        return {
            "customer_id": "unknown",
            "source": "unresolved",
            "match_confidence": 0.0,
            "profile": None,
        }

    def build_preference_vector(self, profile: dict[str, Any]) -> list[tuple[str, float]]:
        interests = profile.get("interests", [])
        vector = [(f"{item['category']}:{item['entity']}", item["weight"]) for item in interests]
        if profile.get("favorite_driver"):
            vector.append((f"driver:{profile['favorite_driver']}", 0.9))
        if profile.get("favorite_brand"):
            vector.append((f"brand:{profile['favorite_brand']}", 0.88))
        return vector


def _time_of_day() -> str:
    hour = datetime.now(timezone.utc).hour
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


peoplecloud = PeopleCloudMock()
