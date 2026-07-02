from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from ..config import settings
from ..store import store
from .confidence_model import confidence_model
from .diagnosis import diagnosis_model
from .intent_model import intent_model
from .nba_policy import nba_policy
from .pipeline import pipeline
from .similarity import build_index


def _days_since_last_conversion(customer_id: str) -> float:
    try:
        events = store.list_events(customer_id)
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


def _collect_hitl_feedback(recent_cutoff_minutes: float) -> None:
    label_map = {"approved": 2, "rejected": 0, "edited": 1}
    for customer_id, history in store.action_history.items():
        for entry in history:
            status = entry.get("status")
            if status not in label_map or entry.get("confidence_processed"):
                continue
            created_at = entry.get("created_at")
            if created_at and _minutes_ago(created_at) > recent_cutoff_minutes:
                continue
            offer_category = entry.get("offer_category", "general")
            confidence_model.record_feedback(
                intent_score=entry.get("intent_score", 0.5),
                fatigue_score=entry.get("fatigue_score", 0.5),
                fatigue_type=entry.get("fatigue_type", "none"),
                channel=entry.get("channel", "email"),
                days_since_last_conversion=entry.get(
                    "days_since_last_conversion",
                    _days_since_last_conversion(customer_id),
                ),
                hitl_override_rate=confidence_model._compute_hitl_override_rate(
                    customer_id, store
                ),
                marketer_reject_rate=confidence_model._compute_marketer_reject_rate(
                    offer_category, store
                ),
                label=label_map[status],
            )
            entry["confidence_processed"] = True


class RetrainScheduler:
    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._running = False
        self._recent_events_buffer: list[dict] = []

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        interval = settings.retrain_interval_minutes * 60
        while self._running:
            await asyncio.sleep(interval)
            await self.run_mini_batch()

    async def run_mini_batch(self) -> dict[str, Any]:
        """Collect recent events/outcomes and retrain all three models."""
        now = datetime.now(timezone.utc)
        customers = store.list_customers()
        events = store.list_events()
        products = store.list_products()
        outcomes = store.list_outcomes()

        recent_cutoff = settings.retrain_interval_minutes
        recent_events = [
            e for e in events
            if _minutes_ago(e["created_at"]) <= recent_cutoff
        ]
        self._recent_events_buffer.extend(recent_events)

        intent_result = intent_model.train(customers, products, events, outcomes)
        diagnosis_result = diagnosis_model.train(customers, events, outcomes)
        nba_result = nba_policy.train(customers, outcomes, events)
        index_result = build_index(products, outcomes)

        try:
            _collect_hitl_feedback(recent_cutoff)
        except Exception:
            pass

        store.model_status.update({
            "intent_model": {
                "version": intent_model.version,
                "last_trained": now.isoformat(),
                "samples": intent_result.get("samples", 0),
            },
            "diagnosis_model": {
                "version": diagnosis_model.version,
                "last_trained": now.isoformat(),
                "samples": diagnosis_result.get("samples", 0),
            },
            "nba_model": {
                "version": nba_policy.version,
                "last_trained": now.isoformat(),
                "samples": nba_result.get("samples", 0),
            },
            "confidence_model": {
                "version": confidence_model.version,
                "last_trained": now.isoformat(),
                "samples": len(confidence_model.feedback_buffer),
            },
            "faiss_index": index_result,
            "retrain_count": store.model_status.get("retrain_count", 0) + 1,
            "last_retrain": now.isoformat(),
            "recent_events_processed": len(recent_events),
        })

        for customer in customers[:50]:
            rec = pipeline.build_recommendation(customer)
            store.upsert_recommendation(rec)

        metrics = pipeline.compute_dashboard_metrics()
        store.log_metrics({**metrics["metrics"], "timestamp": now.isoformat()})
        store.save_runtime_snapshot()

        return {
            "status": "retrained",
            "timestamp": now.isoformat(),
            "intent": intent_result,
            "diagnosis": diagnosis_result,
            "nba": nba_result,
            "index": index_result,
            "recommendations_refreshed": min(50, len(customers)),
        }


def _minutes_ago(iso_ts: str) -> float:
    created = datetime.fromisoformat(iso_ts)
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - created).total_seconds() / 60


retrain_scheduler = RetrainScheduler()
