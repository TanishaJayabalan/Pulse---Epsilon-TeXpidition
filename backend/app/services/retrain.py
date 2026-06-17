from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from ..config import settings
from ..store import store
from .diagnosis import diagnosis_model
from .intent_model import intent_model
from .nba_policy import nba_policy
from .pipeline import pipeline
from .similarity import build_index


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
