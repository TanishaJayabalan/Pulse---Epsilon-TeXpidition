from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

FatigueCause = Literal["channel_fatigue", "timing_mismatch", "offer_fatigue", "none"]
NBAAction = Literal[
    "switch_channel_push",
    "switch_channel_paid",
    "backoff_24h",
    "suppress_7d",
    "change_offer_category",
    "send_default",
]
OutcomeLabel = Literal["converted", "clicked", "ignored", "unsubscribed", "neutral"]


class Interest(BaseModel):
    category: str
    entity: str
    weight: float = Field(ge=0.0, le=1.0)


class CustomerContext(BaseModel):
    device: str = "desktop"
    time_of_day: str = "morning"


class CustomerProfile(BaseModel):
    customer_id: str
    name: str
    interests: list[Interest]
    favorite_driver: str | None = None
    favorite_brand: str | None = None
    fatigue_score: float = 0.0
    context: CustomerContext = Field(default_factory=CustomerContext)
    preferred_channel: str = "email"
    segment: str = "active"
    lifetime_value: float = 0.0
    match_confidence: float = 0.9
    consent: bool = True
    brand: str = "Epsilon Demo"


class BehavioralEvent(BaseModel):
    event_id: str
    customer_id: str
    type: str
    channel: str
    product_id: str
    product_name: str
    offer_category: str
    sentiment: str
    created_at: str
    hour_of_day: int = 10


class Product(BaseModel):
    product_id: str
    name: str
    category: str
    entity: str
    description: str
    price: float
    brand: str


class OutcomeRecord(BaseModel):
    outcome_id: str
    customer_id: str
    product_id: str
    channel: str
    action_taken: str
    label: OutcomeLabel
    created_at: str


class OverridePayload(BaseModel):
    status: str
    reason: str | None = None
    edited_action: str | None = None
    edited_channel: str | None = None
    edited_timing: str | None = None


class SimEventPayload(BaseModel):
    customer_id: str
    type: str
    channel: str
    product: str
    sentiment: str
    offer_category: str = "general"


class RecommendationResponse(BaseModel):
    id: str
    customer_id: str
    customer_name: str
    brand: str
    action: str
    channel: str
    timing: str
    product: str
    product_id: str
    intent_score: float
    fatigue_score: float
    status: str
    diagnosis: str
    fatigue_cause: str
    explanation: str
    top_products: list[dict[str, Any]]
    suppression: bool
    decision_source: str
    nba_action: str
    model_version: str
    confidence_zone: str | None = None
    confidence_score: float | None = None
    confidence_color: str | None = None
    reasoning_summary: str | None = None
