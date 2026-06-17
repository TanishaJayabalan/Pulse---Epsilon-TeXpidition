from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from ..config import MODELS_DIR
from ..store import store
from .diagnosis import diagnosis_model
from .fatigue import detect_fatigue_signals
from .identity import peoplecloud
from .intent_model import intent_model
from .nba_policy import nba_policy
from .similarity import build_index, explain_recommendation, search_top_k


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

    def build_recommendation(self, customer: dict[str, Any]) -> dict[str, Any]:
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

        return {
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
                "pending_reviews": len([r for r in recs if r.get("status") == "pending"]),
            },
            "channel_performance": channel_performance,
            "model_status": store.model_status,
        }


pipeline = PersonalizationPipeline()
