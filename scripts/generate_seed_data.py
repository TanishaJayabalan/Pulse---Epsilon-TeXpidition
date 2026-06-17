"""Generate synthetic seed data for the closed-loop personalization system."""

from __future__ import annotations

import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

INTEREST_POOL = [
    ("sports", "Ferrari", "Charles Leclerc"),
    ("sports", "Red Bull", "Max Verstappen"),
    ("sports", "Mercedes", "Lewis Hamilton"),
    ("food", "Pizza", None),
    ("food", "Sushi", None),
    ("food", "Tacos", None),
    ("music", "Rock", None),
    ("music", "Jazz", None),
    ("music", "EDM", None),
    ("beauty", "Sephora", None),
    ("beauty", "MAC", None),
    ("fashion", "Nike", None),
    ("fashion", "Adidas", None),
    ("tech", "Apple", None),
    ("tech", "Samsung", None),
]

PRODUCT_CATALOG = [
    ("prod_ferrari_cap", "Ferrari Team Cap", "sports", "Ferrari", "Official Ferrari F1 team cap for race day fans.", 45.0, "Ferrari"),
    ("prod_leclerc_poster", "Leclerc Signed Poster", "sports", "Charles Leclerc", "Limited edition Charles Leclerc poster.", 29.0, "F1 Store"),
    ("prod_verstappen_hoodie", "Verstappen Champion Hoodie", "sports", "Max Verstappen", "Max Verstappen championship hoodie.", 79.0, "Red Bull"),
    ("prod_pizza_kit", "Artisan Pizza Kit", "food", "Pizza", "Make restaurant-quality pizza at home.", 34.0, "FoodCraft"),
    ("prod_sushi_set", "Sushi Starter Set", "food", "Sushi", "Premium sushi rolling kit with guide.", 42.0, "FoodCraft"),
    ("prod_rock_vinyl", "Classic Rock Vinyl Bundle", "music", "Rock", "Curated rock vinyl collection.", 55.0, "SoundWave"),
    ("prod_edm_festival", "EDM Festival Pass", "music", "EDM", "VIP access to summer EDM festival.", 120.0, "BeatLive"),
    ("prod_sephora_lip", "Sephora Lip Collection", "beauty", "Sephora", "Top-rated lip color set from Sephora.", 38.0, "Sephora"),
    ("prod_mac_palette", "MAC Eyeshadow Palette", "beauty", "MAC", "Professional MAC eyeshadow palette.", 52.0, "MAC"),
    ("prod_nike_shoes", "Nike Running Shoes", "fashion", "Nike", "Lightweight Nike running shoes.", 110.0, "Nike"),
    ("prod_adidas_track", "Adidas Track Jacket", "fashion", "Adidas", "Classic Adidas track jacket.", 85.0, "Adidas"),
    ("prod_iphone_case", "iPhone Pro Case", "tech", "Apple", "Slim protective case for iPhone Pro.", 49.0, "Apple"),
    ("prod_galaxy_buds", "Galaxy Buds Pro", "tech", "Samsung", "Noise-cancelling Galaxy Buds.", 149.0, "Samsung"),
    ("prod_ferrari_model", "Ferrari Model Car 1:18", "sports", "Ferrari", "Die-cast Ferrari model car.", 65.0, "Ferrari"),
    ("prod_pizza_oven", "Mini Pizza Oven", "food", "Pizza", "Countertop pizza oven for enthusiasts.", 199.0, "FoodCraft"),
]

CHANNELS = ["email", "push", "paid", "onsite"]
EVENT_TYPES_POS = ["click", "cart_add", "purchase", "search", "view", "email_open"]
EVENT_TYPES_NEG = ["ignore", "email_ignore", "ad_skip", "unsubscribe"]
DEVICES = ["mobile", "desktop", "tablet"]
TIMES_OF_DAY = ["morning", "afternoon", "evening", "night"]

NBA_ACTIONS = [
    "switch_channel_push",
    "switch_channel_paid",
    "backoff_24h",
    "suppress_7d",
    "change_offer_category",
    "send_default",
]

FATIGUE_CAUSES = ["channel_fatigue", "timing_mismatch", "offer_fatigue", "none"]


def _uuid() -> str:
    return str(uuid.uuid4())


def generate_customer() -> dict:
    num_interests = random.randint(1, 4)
    picks = random.sample(INTEREST_POOL, num_interests)
    interests = []
    favorite_driver = None
    favorite_brand = None
    for category, entity, driver in picks:
        weight = round(random.uniform(0.55, 0.98), 2)
        interests.append({"category": category, "entity": entity, "weight": weight})
        if driver:
            favorite_driver = driver
        if category == "beauty":
            favorite_brand = entity

    name_parts = ["Ava", "Noah", "Mia", "Liam", "Zoe", "Ethan", "Luna", "Kai", "Iris", "Omar"]
    return {
        "customer_id": _uuid(),
        "name": f"{random.choice(name_parts)} {random.choice(['Chen', 'Patel', 'Garcia', 'Kim', 'Brooks'])}",
        "interests": interests,
        "favorite_driver": favorite_driver,
        "favorite_brand": favorite_brand or random.choice(["Sephora", "Nike", "Apple", None]),
        "fatigue_score": round(random.uniform(0.05, 0.45), 2),
        "context": {
            "device": random.choice(DEVICES),
            "time_of_day": random.choice(TIMES_OF_DAY),
        },
        "preferred_channel": random.choice(CHANNELS),
        "segment": random.choice(["vip", "active", "at_risk", "new"]),
        "lifetime_value": round(random.uniform(120, 4200), 2),
        "match_confidence": round(random.uniform(0.72, 0.99), 2),
        "consent": random.random() > 0.05,
        "brand": random.choice(["Epsilon Demo", "Sephora", "Nike", "F1 Store"]),
    }


def generate_events(customers: list[dict], events_per_customer: tuple[int, int]) -> list[dict]:
    events = []
    now = datetime.now(timezone.utc)
    products = [
        {"product_id": p[0], "name": p[1], "category": p[2], "entity": p[3]}
        for p in PRODUCT_CATALOG
    ]

    for customer in customers:
        count = random.randint(*events_per_customer)
        channel_state: dict[str, int] = {ch: 0 for ch in CHANNELS}
        offer_ignore: dict[str, int] = {}
        morning_opens = 0

        for idx in range(count):
            days_ago = random.randint(0, 30)
            hours_ago = random.randint(0, 23)
            ts = now - timedelta(days=days_ago, hours=hours_ago)
            hour = ts.hour
            channel = random.choice(CHANNELS)
            product = random.choice(products)

            if channel_state[channel] >= 5 and channel == "email":
                event_type = "email_ignore"
                sentiment = "negative"
            elif offer_ignore.get(product["product_id"], 0) >= 3:
                event_type = "ignore"
                sentiment = "negative"
            elif 9 <= hour <= 11 and morning_opens == 0 and days_ago < 5:
                event_type = random.choice(["email_ignore", "ignore"])
                sentiment = "negative"
            else:
                event_type = random.choices(
                    EVENT_TYPES_POS + EVENT_TYPES_NEG,
                    weights=[3, 2, 1, 2, 4, 2, 2, 3, 2, 1],
                    k=1,
                )[0]
                sentiment = "negative" if event_type in EVENT_TYPES_NEG else "positive"

            if event_type == "email_open" and 9 <= hour <= 11:
                morning_opens += 1
            if sentiment == "negative":
                channel_state[channel] = channel_state.get(channel, 0) + 1
                if event_type in ("ignore", "email_ignore"):
                    offer_ignore[product["product_id"]] = offer_ignore.get(product["product_id"], 0) + 1
            else:
                channel_state[channel] = 0
                offer_ignore[product["product_id"]] = 0

            events.append({
                "event_id": _uuid(),
                "customer_id": customer["customer_id"],
                "type": event_type,
                "channel": channel,
                "product_id": product["product_id"],
                "product_name": product["name"],
                "offer_category": product["category"],
                "sentiment": sentiment,
                "created_at": ts.isoformat(),
                "hour_of_day": hour,
            })
    return events


def generate_outcomes(customers: list[dict], events: list[dict]) -> list[dict]:
    outcomes = []
    now = datetime.now(timezone.utc)
    event_sample = random.sample(events, min(len(events), len(customers) * 3))

    for event in event_sample:
        if event["sentiment"] == "positive":
            label = random.choices(["converted", "clicked", "neutral"], weights=[2, 4, 3])[0]
        else:
            label = random.choices(["ignored", "unsubscribed", "neutral"], weights=[6, 1, 2])[0]

        cause = random.choice(FATIGUE_CAUSES)
        action = NBA_ACTIONS[FATIGUE_CAUSES.index(cause)] if cause != "none" else "send_default"
        if cause == "channel_fatigue":
            action = random.choice(["switch_channel_push", "switch_channel_paid"])
        elif cause == "timing_mismatch":
            action = "backoff_24h"
        elif cause == "offer_fatigue":
            action = random.choice(["suppress_7d", "change_offer_category"])

        success = label in ("converted", "clicked")
        outcomes.append({
            "outcome_id": _uuid(),
            "customer_id": event["customer_id"],
            "product_id": event["product_id"],
            "channel": event["channel"],
            "action_taken": action,
            "label": label,
            "fatigue_cause": cause,
            "interaction_success": int(success),
            "created_at": (now - timedelta(hours=random.randint(1, 720))).isoformat(),
        })
    return outcomes


def split_customers(customers: list[dict], train_ratio: float = 0.8) -> tuple[list, list]:
    shuffled = customers.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def main() -> None:
    print("Generating seed data...")
    all_customers = [generate_customer() for _ in range(2500)]
    train_customers, test_customers = split_customers(all_customers, 0.8)

    train_ids = {c["customer_id"] for c in train_customers}
    test_ids = {c["customer_id"] for c in test_customers}

    all_events = generate_events(all_customers, (8, 25))
    train_events = [e for e in all_events if e["customer_id"] in train_ids]
    test_events = [e for e in all_events if e["customer_id"] in test_ids]

    products = [
        {
            "product_id": p[0],
            "name": p[1],
            "category": p[2],
            "entity": p[3],
            "description": p[4],
            "price": p[5],
            "brand": p[6],
        }
        for p in PRODUCT_CATALOG
    ]

    outcomes = generate_outcomes(train_customers, train_events)
    test_outcomes = generate_outcomes(test_customers, test_events)

    files = {
        "customers_train.json": train_customers,
        "customers_test.json": test_customers,
        "events_train.json": train_events,
        "events_test.json": test_events,
        "products.json": products,
        "outcomes_train.json": outcomes,
        "outcomes_test.json": test_outcomes,
    }

    for filename, data in files.items():
        path = DATA_DIR / filename
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        print(f"  Wrote {path} ({len(data)} records)")

    print("Done.")


if __name__ == "__main__":
    main()
