from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .models.schemas import OverridePayload, SimEventPayload
from .services.pipeline import pipeline
from .services.retrain import retrain_scheduler
from .store import store

app = FastAPI(title="Pulse Personalization API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.frontend_origin,
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    store.load_seed()
    asyncio.create_task(_background_init())


async def _background_init() -> None:
    result = await pipeline.warm_start()
    await retrain_scheduler.start()
    store.log_metrics({
        **pipeline.compute_dashboard_metrics()["metrics"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "startup",
        "init": result,
    })


@app.on_event("shutdown")
async def shutdown() -> None:
    await retrain_scheduler.stop()
    store.save_runtime_snapshot()


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "ready": pipeline._initialized,
        "customers": len(store.list_customers()),
        "events": len(store.list_events()),
        "models": store.model_status,
    }


@app.post("/api/seed")
async def seed() -> dict[str, Any]:
    store.reset()
    result = pipeline.initialize()
    return {"status": "seeded", **result}


@app.get("/api/dashboard")
async def dashboard() -> dict[str, Any]:
    if not pipeline._initialized:
        return {
            "store": "PeopleCloud Mock + In-Memory Pipeline",
            "metrics": {"events_per_day": len(store.list_events()), "pending_reviews": 0},
            "channel_performance": [],
            "recommendations": [],
            "model_status": store.model_status,
            "metrics_history": [],
            "warming": True,
        }
    metrics = pipeline.compute_dashboard_metrics()
    recs = store.list_recommendations()
    return {
        "store": "PeopleCloud Mock + In-Memory Pipeline",
        **metrics,
        "recommendations": recs[:100],
        "metrics_history": store.metrics_log[-20:],
        "warming": False,
    }


@app.get("/api/customers")
async def customers() -> list[dict[str, Any]]:
    rows = []
    for customer in store.list_customers()[:200]:
        events = store.list_events(customer["customer_id"])
        rec = store.get_recommendation(f"rec_{customer['customer_id']}")
        rows.append({
            **customer,
            "event_count": len(events),
            "intent_score": rec.get("intent_score", 0) if rec else 0,
            "fatigue_score": rec.get("fatigue_score", 0) if rec else 0,
            "diagnosis": rec.get("diagnosis", "") if rec else "",
            "fatigue_cause": rec.get("fatigue_cause", "none") if rec else "none",
        })
    return rows


@app.get("/api/customers/{customer_id}")
async def customer_detail(customer_id: str) -> dict[str, Any]:
    customer = store.get_customer(customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    events = sorted(
        store.list_events(customer_id),
        key=lambda e: e["created_at"],
        reverse=True,
    )
    rec = pipeline.build_recommendation(customer)
    store.upsert_recommendation(rec)

    return {
        "customer": customer,
        "events": events[:50],
        "recommendation": rec,
        "action_history": store.get_action_history(customer_id),
    }


@app.get("/api/recommendations")
async def recommendations() -> list[dict[str, Any]]:
    if not store.list_recommendations():
        for customer in store.list_customers()[:100]:
            rec = pipeline.build_recommendation(customer)
            store.upsert_recommendation(rec)
    return store.list_recommendations()[:100]


@app.post("/api/recommendations/{recommendation_id}/override")
async def override_recommendation(
    recommendation_id: str,
    payload: OverridePayload,
) -> dict[str, Any]:
    rec = store.get_recommendation(recommendation_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Recommendation not found")

    updated = {
        **rec,
        "status": payload.status,
        "action": payload.edited_action or rec["action"],
        "channel": payload.edited_channel or rec["channel"],
        "timing": payload.edited_timing or rec["timing"],
    }
    store.upsert_recommendation(updated)

    audit = {
        "id": f"ovr_{recommendation_id}_{int(datetime.now(timezone.utc).timestamp())}",
        "recommendation_id": recommendation_id,
        "status": payload.status,
        "reason": payload.reason,
        "delta": {
            "action": [rec["action"], updated["action"]],
            "channel": [rec["channel"], updated["channel"]],
            "timing": [rec["timing"], updated["timing"]],
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    store.add_override(audit)
    return {"recommendation": updated, "audit": audit}


@app.post("/api/simulate")
async def simulate(payload: SimEventPayload) -> dict[str, Any]:
    customer = store.get_customer(payload.customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    products = store.list_products()
    product = next(
        (p for p in products if p["name"].lower() == payload.product.lower()),
        products[0] if products else {"product_id": "unknown", "name": payload.product, "category": payload.offer_category},
    )

    now = datetime.now(timezone.utc)
    event = {
        "event_id": str(uuid.uuid4()),
        "customer_id": payload.customer_id,
        "type": payload.type,
        "channel": payload.channel,
        "product_id": product["product_id"],
        "product_name": product.get("name", payload.product),
        "offer_category": payload.offer_category or product.get("category", "general"),
        "sentiment": payload.sentiment,
        "created_at": now.isoformat(),
        "hour_of_day": now.hour,
    }

    result = pipeline.process_event(event)
    rec = result["recommendation"]

    label = "clicked" if payload.sentiment == "positive" else "ignored"
    if payload.type == "purchase":
        label = "converted"
    if payload.type == "unsubscribe":
        label = "unsubscribed"

    outcome = pipeline.capture_outcome(
        payload.customer_id,
        product["product_id"],
        payload.channel,
        rec.get("nba_action", "send_default"),
        label,
    )

    return {**result, "outcome": outcome}


@app.post("/api/retrain")
async def trigger_retrain() -> dict[str, Any]:
    return await retrain_scheduler.run_mini_batch()


@app.get("/api/models/status")
async def model_status() -> dict[str, Any]:
    return store.model_status


@app.get("/api/products")
async def products() -> list[dict[str, Any]]:
    return store.list_products()
