from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from ..config import MODELS_DIR
from ..store import store
from .confidence_model import confidence_model
from .diagnosis import diagnosis_model
from .fatigue import detect_fatigue_signals
from .identity import peoplecloud
from .intent_model import intent_model
from .llm_reasoning import llm_reasoning
from .nba_policy import nba_policy
from .similarity import build_index, explain_recommendation, search_top_k


def _days_since_last_conversion(events: list[dict[str, Any]]) -> float:
    try:
        for event in sorted(events, key=lambda item: item.get("created_at", ""), reverse=True):
            if event.get("type") in ("purchase", "converted"):
                created = datetime.fromisoformat(event["created_at"].replace("Z", "+00:00"))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                delta_days = (datetime.now(timezone.utc) - created).days
                return float(min(30, max(0, delta_days)))
        return 30.0
    except Exception:
        return 30.0


HITL_LOCKED_STATUSES = frozenset({"approved", "rejected", "edited"})


def _apply_confidence_zone(
    recommendation: dict[str, Any],
    status: str,
    intent_score: float,
    fatigue: dict[str, Any],
    cause: str,
) -> None:
    """Align status, display fields, and queue eligibility with confidence zone."""
    zone = recommendation.get("confidence_zone", "NEEDS_REVIEW")
    intent_pct = round(intent_score * 100, 1)
    fatigue_pct = recommendation.get("fatigue_score", 0)

    if zone == "AUTO_SUPPRESSED":
        reason_str = recommendation.get("suppression_reason", "high model rejection risk")
        recommendation["effective_action"] = f"Hold outreach — auto-suppressed due to {reason_str}"
        recommendation["effective_channel"] = "none"
        recommendation["queue_eligible"] = False
        if status not in HITL_LOCKED_STATUSES:
            recommendation["status"] = "auto_suppressed"
    elif zone == "AUTO_EXECUTE":
        recommendation["effective_action"] = recommendation["action"]
        recommendation["effective_channel"] = recommendation["channel"]
        recommendation["queue_eligible"] = False
        if status not in HITL_LOCKED_STATUSES:
            recommendation["status"] = "auto_executed"
    else:
        recommendation["effective_action"] = recommendation["action"]
        recommendation["effective_channel"] = recommendation["channel"]
        recommendation["queue_eligible"] = True
        if status not in HITL_LOCKED_STATUSES:
            recommendation["status"] = "pending"

    if status in HITL_LOCKED_STATUSES:
        recommendation["status"] = status

    recommendation["zone_rationale"] = {
        "AUTO_SUPPRESSED": (
            f"Intent {intent_pct}% is below the safe send threshold"
            + (
                f" and fatigue is elevated at {fatigue_pct}% ({cause.replace('_', ' ')})"
                if fatigue_pct >= 50 or cause not in ("none",)
                else ""
            )
            + " — outreach held back automatically."
        ),
        "AUTO_EXECUTE": (
            f"Strong intent ({intent_pct}%), low fatigue ({fatigue_pct}%), "
            "and stable HITL history — safe to auto-execute."
        ),
        "NEEDS_REVIEW": (
            f"Mixed signals (intent {intent_pct}%, fatigue {fatigue_pct}%, {cause.replace('_', ' ')}) "
            "— marketer review required before send."
        ),
    }.get(zone, "")


def _recommendation_is_stale(rec: dict[str, Any]) -> bool:
    zone = rec.get("confidence_zone")
    status = rec.get("status")
    if not zone or rec.get("queue_eligible") is None:
        return True
    if zone == "AUTO_SUPPRESSED" and status == "pending":
        return True
    if zone == "NEEDS_REVIEW" and status == "auto_suppressed":
        return True
    if zone == "AUTO_EXECUTE" and status == "pending":
        return True
    if zone == "NEEDS_REVIEW" and status not in ("pending", *HITL_LOCKED_STATUSES):
        return True
    return False


def _models_exist() -> bool:
    return (
        (MODELS_DIR / "intent_model.txt").exists()
        and (MODELS_DIR / "diagnosis_model.joblib").exists()
        and (MODELS_DIR / "nba_model.joblib").exists()
        and (MODELS_DIR / "product_index.faiss").exists()
    )


class PersonalizationPipeline:
    def __init__(self) -> None:
        self._initialized = False
        self._warming = False

    def initialize(self, *, force_retrain: bool = False) -> dict[str, Any]:
        if self._initialized and not force_retrain:
            return {"status": "already_initialized"}

        store.load_seed()
        products = store.list_products()
        outcomes = store.list_outcomes()

        if _models_exist() and not force_retrain:
            index_result = build_index(products, outcomes)
            intent_result = {"status": "loaded", "version": intent_model.version}
            diagnosis_result = {"status": "loaded", "version": diagnosis_model.version}
            nba_result = {"status": "loaded", "version": nba_policy.version}
        else:
            index_result = build_index(products, outcomes)
            intent_result = intent_model.train(
                store.list_customers(), products, store.list_events(), outcomes
            )
            diagnosis_result = diagnosis_model.train(
                store.list_customers(), store.list_events(), outcomes
            )
            nba_result = nba_policy.train(
                store.list_customers(), outcomes, store.list_events()
            )

        store.model_status = {
            "intent_model": {
                "version": intent_model.version,
                "last_trained": datetime.now(timezone.utc).isoformat(),
                "samples": intent_result.get("samples", 0),
            },
            "diagnosis_model": {
                "version": diagnosis_model.version,
                "last_trained": datetime.now(timezone.utc).isoformat(),
                "samples": diagnosis_result.get("samples", 0),
            },
            "nba_model": {
                "version": nba_policy.version,
                "last_trained": datetime.now(timezone.utc).isoformat(),
                "samples": nba_result.get("samples", 0),
            },
            "faiss_index": index_result,
            "retrain_count": 0,
            "last_retrain": datetime.now(timezone.utc).isoformat(),
        }

        for customer in store.list_customers()[:100]:
            rec = self.build_recommendation(customer)
            store.upsert_recommendation(rec)

        self._initialized = True
        return {
            "status": "initialized",
            "customers": len(store.list_customers()),
            "events": len(store.list_events()),
            "products": len(products),
            "models": store.model_status,
            "cached": _models_exist() and not force_retrain,
        }

    async def warm_start(self) -> dict[str, Any]:
        if self._warming:
            return {"status": "warming"}
        self._warming = True
        try:
            return await asyncio.to_thread(self.initialize)
        finally:
            self._warming = False

    def build_recommendation(
        self,
        customer: dict[str, Any],
        *,
        include_reasoning: bool = False,
    ) -> dict[str, Any]:
        cid = customer["customer_id"]
        events = store.list_events(cid)
        products = store.list_products()

        identity = peoplecloud.resolve(cid)
        profile = identity.get("profile") or customer

        fatigue = detect_fatigue_signals(events)
        diagnosis = diagnosis_model.predict(profile, events, fatigue)
        action_history = store.get_action_history(cid)

        intent_scored = intent_model.score_all_products(
            profile, products, events, fatigue["fatigue_score"]
        )
        top_products = search_top_k(profile, intent_scored)
        top_product = top_products[0] if top_products else (products[0] if products else {})

        nba = nba_policy.decide(profile, fatigue, diagnosis, action_history)
        delivery = nba_policy.action_to_delivery(nba["action"], profile, top_product)

        intent_score = intent_scored[0][1] if intent_scored else 0.35
        suppression = nba["action"] == "suppress_7d" or fatigue["fatigue_score"] > 0.75

        cause = diagnosis["cause"]
        diagnosis_text = {
            "channel_fatigue": f"Channel fatigue detected: {fatigue['email_ignore_streak']} consecutive email ignores.",
            "timing_mismatch": f"Timing mismatch: no opens between 9–11am for {fatigue['morning_ignore_days']} days.",
            "offer_fatigue": f"Offer fatigue: same offer ignored {fatigue['max_offer_ignores']} times.",
            "none": "No significant fatigue; proceed with intent-based outreach.",
        }.get(cause, "Monitoring signals.")

        model_version = f"i{intent_model.version}-d{diagnosis_model.version}-n{nba_policy.version}"

        existing = store.get_recommendation(f"rec_{cid}")
        status = existing.get("status", "pending") if existing else "pending"

        recommendation = {
            "id": f"rec_{cid}",
            "customer_id": cid,
            "customer_name": customer.get("name", "Unknown"),
            "brand": customer.get("brand", "Epsilon Demo"),
            "action": delivery["action_label"],
            "channel": delivery["channel"],
            "timing": delivery["timing"],
            "product": top_product.get("name", "Recommended offer"),
            "product_id": top_product.get("product_id", ""),
            "intent_score": round(intent_score * 100, 1),
            "fatigue_score": round(fatigue["fatigue_score"] * 100, 1),
            "status": status,
            "diagnosis": diagnosis_text,
            "fatigue_cause": cause,
            "explanation": explain_recommendation(profile, top_products),
            "top_products": top_products,
            "suppression": suppression,
            "decision_source": nba["source"],
            "nba_action": nba["action"],
            "nba_strategy": nba["strategy"],
            "nba_reason": nba["reason"],
            "model_version": model_version,
            "channel_health": fatigue.get("channel_consecutive", {}),
            "identity_source": identity["source"],
            "match_confidence": identity["match_confidence"],
        }

        try:
            offer_category = top_product.get("category", "general")
            confidence_result = confidence_model.predict(
                intent_score=intent_score,
                fatigue_score=fatigue["fatigue_score"],
                fatigue_type=cause,
                channel=delivery["channel"],
                days_since_last_conversion=_days_since_last_conversion(events),
                customer_id=cid,
                offer_category=offer_category,
                store=store,
                nba_suppressed=nba["action"] == "suppress_7d",
            )
            recommendation["confidence_zone"] = confidence_result["zone"]
            recommendation["confidence_score"] = confidence_result["confidence"]
            recommendation["confidence_color"] = confidence_result["color"]
            if "suppression_reason" in confidence_result:
                recommendation["suppression_reason"] = confidence_result["suppression_reason"]
        except Exception:
            recommendation["confidence_zone"] = "NEEDS_REVIEW"
            recommendation["confidence_score"] = 0.5
            recommendation["confidence_color"] = "amber"

        _apply_confidence_zone(recommendation, status, intent_score, fatigue, cause)

        existing_summary = existing.get("reasoning_summary") if existing else None
        existing_fingerprint = existing.get("reasoning_fingerprint") if existing else None
        current_fingerprint = llm_reasoning.recommendation_fingerprint(recommendation)

        if existing_summary and existing_fingerprint == current_fingerprint:
            recommendation["reasoning_summary"] = existing_summary
            recommendation["reasoning_fingerprint"] = existing_fingerprint
        elif include_reasoning:
            try:
                recommendation["reasoning_summary"] = llm_reasoning.generate_reasoning(
                    customer=profile,
                    recommendation=recommendation,
                    fatigue_result=fatigue,
                    intent_score=intent_score,
                    events=events,
                )
                llm_reasoning.cache_summary(cid, recommendation, recommendation["reasoning_summary"])
            except Exception:
                recommendation["reasoning_summary"] = (
                    "Recommendation generated based on behavioral signal analysis."
                )

        return recommendation

    def refresh_stale_recommendations(self, limit: int = 100, *, force_all: bool = False) -> int:
        """Rebuild cached recommendations so queue and customer detail stay in sync."""
        refreshed = 0
        for rec in store.list_recommendations():
            if refreshed >= limit:
                break
            if not force_all and not _recommendation_is_stale(rec):
                continue
            customer = store.get_customer(rec["customer_id"])
            if not customer:
                continue
            existing = store.get_recommendation(rec["id"])
            fresh = self.build_recommendation(customer, include_reasoning=False)
            if existing and existing.get("reasoning_summary"):
                fp = llm_reasoning.recommendation_fingerprint(fresh)
                if existing.get("reasoning_fingerprint") == fp:
                    fresh["reasoning_summary"] = existing["reasoning_summary"]
                    fresh["reasoning_fingerprint"] = fp
            store.upsert_recommendation(fresh)
            refreshed += 1
        return refreshed

    def attach_reasoning(
        self,
        customer: dict[str, Any],
        recommendation: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate LLM reasoning only when viewing a customer detail page."""
        cid = customer["customer_id"]
        cached = llm_reasoning.get_cached(cid, recommendation)
        if cached:
            recommendation["reasoning_summary"] = cached
            return recommendation

        events = store.list_events(cid)
        fatigue = detect_fatigue_signals(events)
        identity = peoplecloud.resolve(cid)
        profile = identity.get("profile") or customer
        intent_score = recommendation.get("intent_score", 35)
        if isinstance(intent_score, (int, float)) and intent_score > 1:
            intent_score = intent_score / 100.0

        try:
            recommendation["reasoning_summary"] = llm_reasoning.generate_reasoning(
                customer=profile,
                recommendation=recommendation,
                fatigue_result=fatigue,
                intent_score=float(intent_score),
                events=events,
            )
            llm_reasoning.cache_summary(cid, recommendation, recommendation["reasoning_summary"])
        except Exception:
            recommendation["reasoning_summary"] = (
                "Recommendation generated based on behavioral signal analysis."
            )
        return recommendation

    def process_event(self, event: dict[str, Any]) -> dict[str, Any]:
        store.add_event(event)
        customer = store.get_customer(event["customer_id"])
        if not customer:
            return {"error": "customer not found"}

        rec = self.build_recommendation(customer)
        store.upsert_recommendation(rec)
        return {"event": event, "recommendation": rec}

    def capture_outcome(
        self,
        customer_id: str,
        product_id: str,
        channel: str,
        action_taken: str,
        label: str,
    ) -> dict[str, Any]:
        import uuid

        outcome = {
            "outcome_id": str(uuid.uuid4()),
            "customer_id": customer_id,
            "product_id": product_id,
            "channel": channel,
            "action_taken": action_taken,
            "label": label,
            "interaction_success": int(label in ("converted", "clicked")),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        store.add_outcome(outcome)
        store.add_action_history(customer_id, outcome)
        return outcome

    def compute_dashboard_metrics(self) -> dict[str, Any]:
        customers = store.list_customers()
        events = store.list_events()
        recs = store.list_recommendations()
        overrides = store.list_overrides()
        outcomes = store.list_outcomes()

        suppressed = [r for r in recs if r.get("suppression")]
        positive_outcomes = [o for o in outcomes if o.get("label") in ("converted", "clicked")]
        interaction_rate = (
            len(positive_outcomes) / max(1, len(outcomes)) * 100
        )

        reentry_count = sum(
            1 for o in outcomes
            if o.get("label") in ("converted", "clicked")
            and any(r.get("suppression") for r in recs if r["customer_id"] == o["customer_id"])
        )

        channel_stats: dict[str, dict] = {}
        for event in events:
            ch = event["channel"]
            channel_stats.setdefault(ch, {"events": 0, "positive": 0, "negative": 0})
            channel_stats[ch]["events"] += 1
            if event.get("sentiment") == "positive":
                channel_stats[ch]["positive"] += 1
            elif event.get("sentiment") == "negative":
                channel_stats[ch]["negative"] += 1

        channel_performance = []
        for ch, stats in channel_stats.items():
            total = stats["positive"] + stats["negative"]
            ctr = round(stats["positive"] / max(1, total) * 100, 1)
            fatigue = round(stats["negative"] / max(1, total) * 100, 1)
            channel_performance.append({"channel": ch, "ctr": ctr, "fatigue": fatigue})

        if not channel_performance:
            channel_performance = [
                {"channel": "email", "ctr": 0, "fatigue": 0},
                {"channel": "push", "ctr": 0, "fatigue": 0},
            ]

        zone_counts = {
            "AUTO_SUPPRESSED": 0,
            "NEEDS_REVIEW": 0,
            "AUTO_EXECUTE": 0,
        }
        for rec in recs:
            zone = rec.get("confidence_zone")
            if zone in zone_counts:
                zone_counts[zone] += 1

        return {
            "metrics": {
                "events_per_day": len(events),
                "id_match_rate": round(
                    sum(c.get("match_confidence", 0.9) for c in customers) / max(1, len(customers)) * 100, 1
                ),
                "suppression_rate": round(len(suppressed) / max(1, len(recs)) * 100, 1),
                "reentry_rate": round(reentry_count / max(1, len(suppressed)) * 100, 1) if suppressed else 0,
                "override_rate": round(len(overrides) / max(1, len(recs)) * 100, 1),
                "interaction_rate": round(interaction_rate, 1),
                "conversion_lift": round(interaction_rate * 0.12, 1),
                "churn_risk_drop": round(len([o for o in outcomes if o.get("label") == "unsubscribed"]) / max(1, len(outcomes)) * 100, 1),
                "approved": len([o for o in overrides if o.get("status") == "approved"]),
                "rejected": len([o for o in overrides if o.get("status") == "rejected"]),
                "pending_reviews": len([
                    r for r in recs
                    if r.get("confidence_zone") == "NEEDS_REVIEW"
                    and r.get("status") == "pending"
                ]),
                "zone_auto_suppressed": zone_counts["AUTO_SUPPRESSED"],
                "zone_needs_review": zone_counts["NEEDS_REVIEW"],
                "zone_auto_execute": zone_counts["AUTO_EXECUTE"],
            },
            "channel_performance": channel_performance,
            "model_status": store.model_status,
        }


pipeline = PersonalizationPipeline()
