from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

FALLBACK_SUMMARY = "Recommendation generated based on behavioral signal analysis."


class LLMReasoning:
    def __init__(self) -> None:
        self.client = None
        self.model = "llama-3.3-70b-versatile"
        self._cache: dict[str, str] = {}
        try:
            from groq import Groq

            api_key = os.environ.get("GROQ_API_KEY", "").strip()
            if api_key and api_key not in ("your_key_here", "changeme", "placeholder"):
                self.client = Groq(api_key=api_key)
        except Exception:
            self.client = None

    @staticmethod
    def recommendation_fingerprint(recommendation: dict[str, Any]) -> str:
        parts = (
            recommendation.get("effective_action") or recommendation.get("action"),
            recommendation.get("effective_channel") or recommendation.get("channel"),
            recommendation.get("product_id"),
            recommendation.get("intent_score"),
            recommendation.get("fatigue_score"),
            recommendation.get("nba_action"),
            recommendation.get("confidence_zone"),
        )
        return "|".join(str(part) for part in parts)

    def get_cached(self, customer_id: str, recommendation: dict[str, Any]) -> str | None:
        fingerprint = self.recommendation_fingerprint(recommendation)
        if (
            recommendation.get("reasoning_summary")
            and recommendation.get("reasoning_fingerprint") == fingerprint
        ):
            return str(recommendation["reasoning_summary"])
        return self._cache.get(f"{customer_id}:{fingerprint}")

    def cache_summary(
        self,
        customer_id: str,
        recommendation: dict[str, Any],
        summary: str,
    ) -> None:
        fingerprint = self.recommendation_fingerprint(recommendation)
        self._cache[f"{customer_id}:{fingerprint}"] = summary
        recommendation["reasoning_fingerprint"] = fingerprint
        recommendation["reasoning_summary"] = summary

    def _channel_history_summary(self, customer_id: str, events: list[dict[str, Any]]) -> str:
        try:
            if not events:
                return "No recent channel activity recorded."
            channel_counts: dict[str, int] = {}
            for event in events[-20:]:
                channel = event.get("channel", "unknown")
                channel_counts[channel] = channel_counts.get(channel, 0) + 1
            parts = [f"{channel}: {count} events" for channel, count in channel_counts.items()]
            return ", ".join(parts) if parts else "No recent channel activity recorded."
        except Exception:
            return "No recent channel activity recorded."

    def _days_since_last_conversion(self, events: list[dict[str, Any]]) -> int:
        try:
            for event in sorted(events, key=lambda item: item.get("created_at", ""), reverse=True):
                if event.get("type") in ("purchase", "converted"):
                    created = event.get("created_at")
                    if not created:
                        continue
                    event_dt = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
                    if event_dt.tzinfo is None:
                        event_dt = event_dt.replace(tzinfo=timezone.utc)
                    now = datetime.now(timezone.utc)
                    return min(30, max(0, (now - event_dt).days))
            return 30
        except Exception:
            return 30

    def _zone_aligned_summary(
        self,
        recommendation: dict[str, Any],
        fatigue_result: dict[str, Any],
        intent_score: float,
    ) -> str:
        zone = recommendation.get("confidence_zone", "NEEDS_REVIEW")
        rationale = recommendation.get("zone_rationale", "")
        fatigue_type = fatigue_result.get(
            "primary_cause", recommendation.get("fatigue_cause", "none")
        ).replace("_", " ")
        intent_pct = round(intent_score * 100 if intent_score <= 1 else intent_score, 1)
        fatigue_pct = recommendation.get("fatigue_score", 0)

        if zone == "AUTO_SUPPRESSED":
            return (
                f"Outreach is suppressed, not recommended for send. "
                f"Intent at {intent_pct}% is too weak to justify another touch "
                f"(fatigue {fatigue_pct}%, {fatigue_type}). "
                f"{rationale}"
            ).strip()
        if zone == "AUTO_EXECUTE":
            action = recommendation.get("effective_action") or recommendation.get("action", "")
            channel = recommendation.get("effective_channel") or recommendation.get("channel", "")
            return (
                f"Signals support automated execution via {channel}: {action}. "
                f"Intent {intent_pct}% and fatigue {fatigue_pct}% are within safe bounds. "
                f"{rationale}"
            ).strip()
        return rationale or FALLBACK_SUMMARY

    def generate_reasoning(
        self,
        customer: dict[str, Any],
        recommendation: dict[str, Any],
        fatigue_result: dict[str, Any],
        intent_score: float,
        events: list[dict[str, Any]] | None = None,
        *,
        force_refresh: bool = False,
    ) -> str:
        try:
            customer_id = customer.get("customer_id", "")
            zone = recommendation.get("confidence_zone", "NEEDS_REVIEW")

            if not force_refresh:
                cached = self.get_cached(customer_id, recommendation)
                if cached:
                    return cached

            if zone in ("AUTO_SUPPRESSED", "AUTO_EXECUTE"):
                summary = self._zone_aligned_summary(recommendation, fatigue_result, intent_score)
                if customer_id:
                    self.cache_summary(customer_id, recommendation, summary)
                return summary

            if not self.client:
                return self._zone_aligned_summary(recommendation, fatigue_result, intent_score)

            interests = ", ".join(
                interest.get("entity", "")
                for interest in customer.get("interests", [])
                if interest.get("entity")
            ) or "general interests"
            fatigue_score = fatigue_result.get("fatigue_score", recommendation.get("fatigue_score", 0))
            if isinstance(fatigue_score, (int, float)) and fatigue_score > 1.0:
                fatigue_score = fatigue_score / 100.0
            fatigue_type = fatigue_result.get(
                "primary_cause", recommendation.get("fatigue_cause", "none")
            )
            days = self._days_since_last_conversion(events or [])
            channel_history = self._channel_history_summary(
                customer.get("customer_id", ""),
                events or [],
            )
            if isinstance(intent_score, (int, float)) and intent_score > 1.0:
                intent_score = intent_score / 100.0

            effective_action = recommendation.get("effective_action") or recommendation.get("action", "")
            effective_channel = recommendation.get("effective_channel") or recommendation.get("channel", "")

            prompt = f"""You are a marketing intelligence assistant for Pulse, a closed-loop \
personalisation engine.

Write a 2-3 sentence reasoning summary for a marketer reviewing this recommendation \
before it is sent. The confidence zone is NEEDS_REVIEW — explain why human review is \
required and what mixed signals the data shows.

Rules:
- Be specific and concise; avoid generic marketing language.
- Do not start with "I" or "This customer". Start with an insight.
- The proposed action still requires marketer approval — explain the tradeoff, not certainty.

Customer profile:
- Interests: {interests}
- Fatigue score: {fatigue_score}
- Fatigue type: {fatigue_type}
- Days since last conversion: {days}
- Channel history: {channel_history}

Proposed action (pending review):
- Action: {effective_action}
- Channel: {effective_channel}
- Product: {recommendation.get("product", "")}
- Intent score: {intent_score}
- Confidence zone: NEEDS_REVIEW
- System note: {recommendation.get("zone_rationale", "")}

Write the reasoning summary now:"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.4,
            )
            content = response.choices[0].message.content
            if content and content.strip():
                summary = content.strip()
                if customer_id:
                    self.cache_summary(customer_id, recommendation, summary)
                return summary
            return self._zone_aligned_summary(recommendation, fatigue_result, intent_score)
        except Exception:
            return self._zone_aligned_summary(recommendation, fatigue_result, intent_score)


llm_reasoning = LLMReasoning()
