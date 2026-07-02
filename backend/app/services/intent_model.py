from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from ..config import MODELS_DIR
from .signals import build_time_decayed_features

FEATURE_COLS = [
    "clicks_24h",
    "clicks_7d",
    "cart_adds_7d",
    "purchases_30d",
    "email_opens_30d",
    "searches_7d",
    "views_7d",
    "negative_7d",
    "product_clicks_7d",
    "product_views_7d",
    "product_cart_7d",
    "interest_match",
    "category_match",
    "channel_pref",
    "fatigue_score",
    "hour_of_day",
]


class IntentModel:
    def __init__(self) -> None:
        self.model: lgb.Booster | None = None
        self.version = 0
        self._load()

    def _interest_match(self, customer: dict, product: dict) -> float:
        for interest in customer.get("interests", []):
            if interest["entity"].lower() in product.get("entity", "").lower():
                return interest["weight"]
            if interest["category"] == product.get("category"):
                return interest["weight"] * 0.7
        return 0.1

    def _build_row(
        self,
        customer: dict,
        product: dict,
        events: list[dict],
        fatigue_score: float = 0.0,
    ) -> dict[str, float]:
        features = build_time_decayed_features(customer["customer_id"], events, product["product_id"])
        return {
            **features,
            "interest_match": self._interest_match(customer, product),
            "category_match": 1.0 if any(
                i["category"] == product.get("category") for i in customer.get("interests", [])
            ) else 0.0,
            "channel_pref": {"email": 0.25, "push": 0.5, "paid": 0.2, "onsite": 0.6}.get(
                customer.get("preferred_channel", "email"), 0.3
            ),
            "fatigue_score": fatigue_score,
            "hour_of_day": datetime.now(timezone.utc).hour / 24.0,
        }

    def train(
        self,
        customers: list[dict],
        products: list[dict],
        events: list[dict],
        outcomes: list[dict],
    ) -> dict[str, Any]:
        rows = []
        labels = []
        product_map = {p["product_id"]: p for p in products}

        positive_outcomes = {
            (o["customer_id"], o["product_id"])
            for o in outcomes
            if o["label"] in ("converted", "clicked")
        }

        for customer in customers[:500]:
            cid = customer["customer_id"]
            cust_events = [e for e in events if e["customer_id"] == cid]
            fatigue = sum(1 for e in cust_events if e.get("sentiment") == "negative") / max(1, len(cust_events))

            sampled_products = products if len(products) <= 8 else [
                product_map[pid] for pid in list(product_map.keys())[:8]
            ]
            for product in sampled_products:
                row = self._build_row(customer, product, events, fatigue)
                rows.append(row)
                labels.append(int((cid, product["product_id"]) in positive_outcomes))

        if len(rows) < 20:
            return {"status": "skipped", "reason": "insufficient data"}

        df = pd.DataFrame(rows)[FEATURE_COLS]
        train_data = lgb.Dataset(df, label=labels)
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "seed": 42,
        }
        self.model = lgb.train(params, train_data, num_boost_round=80)
        self.version += 1
        self._save()
        return {"status": "trained", "version": self.version, "samples": len(rows)}

    def predict(
        self,
        customer: dict,
        product: dict,
        events: list[dict],
        fatigue_score: float = 0.0,
    ) -> float:
        row = self._build_row(customer, product, events, fatigue_score)
        if self.model is None:
            score = (
                row["interest_match"] * 0.4
                + row["product_clicks_7d"] * 0.15
                + row["product_cart_7d"] * 0.2
                + row["purchases_30d"] * 0.15
                - row["negative_7d"] * 0.1
                - fatigue_score * 0.2
            )
            return max(0.0, min(1.0, score))

        vec = np.array([[row[col] for col in FEATURE_COLS]])
        prob = float(self.model.predict(vec)[0])
        
        # Boost probability heuristically for recent strong signals 
        # so the simulator feels highly responsive to injected events.
        if row.get("product_cart_7d", 0) > 0:
            prob += 0.35
        elif row.get("product_clicks_7d", 0) > 0:
            prob += 0.20
        if row.get("purchases_30d", 0) > 0:
            prob += 0.15
            
        return max(0.0, min(1.0, prob))

    def score_all_products(
        self,
        customer: dict,
        products: list[dict],
        events: list[dict],
        fatigue_score: float = 0.0,
    ) -> list[tuple[dict, float]]:
        if not products:
            return []

        rows = [self._build_row(customer, p, events, fatigue_score) for p in products]
        
        if self.model is None:
            scored = []
            for i, p in enumerate(products):
                row = rows[i]
                score = (
                    row["interest_match"] * 0.4
                    + row["product_clicks_7d"] * 0.15
                    + row["product_cart_7d"] * 0.2
                    + row["purchases_30d"] * 0.15
                    - row["negative_7d"] * 0.1
                    - fatigue_score * 0.2
                )
                scored.append((p, max(0.0, min(1.0, score))))
            return sorted(scored, key=lambda x: x[1], reverse=True)

        # Vectorized prediction
        vec = np.array([[row[col] for col in FEATURE_COLS] for row in rows])
        probs = self.model.predict(vec)
        
        scored = []
        for i, p in enumerate(products):
            row = rows[i]
            prob = float(probs[i])
            
            # Boost probability heuristically for recent strong signals 
            # so the simulator feels highly responsive to injected events.
            if row.get("product_cart_7d", 0) > 0:
                prob += 0.35
            elif row.get("product_clicks_7d", 0) > 0:
                prob += 0.20
            if row.get("purchases_30d", 0) > 0:
                prob += 0.15
                
            scored.append((p, max(0.0, min(1.0, prob))))

        return sorted(scored, key=lambda x: x[1], reverse=True)

    def _save(self) -> None:
        if self.model is None:
            return
        self.model.save_model(str(MODELS_DIR / "intent_model.txt"))
        joblib.dump({"version": self.version}, MODELS_DIR / "intent_meta.joblib")

    def _load(self) -> None:
        model_path = MODELS_DIR / "intent_model.txt"
        meta_path = MODELS_DIR / "intent_meta.joblib"
        if model_path.exists():
            self.model = lgb.Booster(model_file=str(model_path))
        if meta_path.exists():
            self.version = joblib.load(meta_path).get("version", 0)


intent_model = IntentModel()
