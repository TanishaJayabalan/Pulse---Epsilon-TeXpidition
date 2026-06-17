from __future__ import annotations

from typing import Any

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from ..config import MODELS_DIR

NBA_ACTIONS = [
    "switch_channel_push",
    "switch_channel_paid",
    "backoff_24h",
    "suppress_7d",
    "change_offer_category",
    "send_default",
]

CAUSE_TO_SINGLE_ACTION = {
    "channel_fatigue": "switch_channel_push",
    "timing_mismatch": "backoff_24h",
    "offer_fatigue": "change_offer_category",
    "none": "send_default",
}


class NBAPolicy:
    def __init__(self) -> None:
        self.model: DecisionTreeClassifier | None = None
        self.version = 0
        self._load()

    def _features(
        self,
        customer: dict,
        fatigue: dict,
        diagnosis: dict,
        action_history: list[dict],
    ) -> list[float]:
        last_success = 0.0
        if action_history:
            last_success = float(action_history[-1].get("interaction_success", 0))

        return [
            fatigue.get("fatigue_score", 0),
            float(diagnosis.get("confidence", 0.5)),
            {"channel_fatigue": 0, "timing_mismatch": 1, "offer_fatigue": 2, "none": 3}.get(
                diagnosis.get("cause", "none"), 3
            ),
            fatigue.get("email_ignore_streak", 0),
            fatigue.get("max_offer_ignores", 0),
            len(action_history),
            last_success,
            {"email": 0, "push": 1, "paid": 2, "onsite": 3}.get(
                customer.get("preferred_channel", "email"), 0
            ),
        ]

    def train(self, customers: list[dict], outcomes: list[dict], events: list[dict]) -> dict[str, Any]:
        from .fatigue import detect_fatigue_signals
        from .diagnosis import diagnosis_model

        X, y = [], []
        customer_map = {c["customer_id"]: c for c in customers}

        for outcome in outcomes:
            cid = outcome["customer_id"]
            customer = customer_map.get(cid)
            if not customer:
                continue
            cust_events = [e for e in events if e["customer_id"] == cid]
            fatigue = detect_fatigue_signals(cust_events)
            diagnosis = diagnosis_model.predict(customer, cust_events, fatigue)
            history = []
            X.append(self._features(customer, fatigue, diagnosis, history))
            y.append(outcome.get("action_taken", "send_default"))

        if len(X) < 20:
            return {"status": "skipped", "reason": "insufficient data"}

        self.model = DecisionTreeClassifier(max_depth=6, random_state=42)
        self.model.fit(np.array(X), y)
        self.version += 1
        self._save()
        return {"status": "trained", "version": self.version, "samples": len(X)}

    def decide(
        self,
        customer: dict,
        fatigue: dict,
        diagnosis: dict,
        action_history: list[dict],
    ) -> dict[str, Any]:
        cause = diagnosis.get("cause", "none")

        if not action_history:
            action = CAUSE_TO_SINGLE_ACTION.get(cause, "send_default")
            return {
                "action": action,
                "strategy": "single_change",
                "source": "heuristic",
                "reason": f"First iteration: apply single change for {cause}",
            }

        last = action_history[-1]
        if last.get("interaction_success", 0) == 0 and self.model is not None:
            vec = np.array([self._features(customer, fatigue, diagnosis, action_history)])
            action = self.model.predict(vec)[0]
            proba = self.model.predict_proba(vec)[0]
            confidence = float(max(proba))
            return {
                "action": action,
                "strategy": "combination",
                "source": "decision_tree",
                "confidence": round(confidence, 3),
                "reason": "Single change did not improve interaction; using decision tree combination",
            }

        if last.get("interaction_success", 0) == 1:
            return {
                "action": last.get("action_taken", "send_default"),
                "strategy": "maintain",
                "source": "history",
                "reason": "Previous action improved interaction rate; maintaining strategy",
            }

        action = CAUSE_TO_SINGLE_ACTION.get(cause, "send_default")
        return {
            "action": action,
            "strategy": "single_change",
            "source": "heuristic",
            "reason": f"Applying single change for {cause}",
        }

    def action_to_delivery(self, action: str, customer: dict, top_product: dict) -> dict[str, str]:
        channel = customer.get("preferred_channel", "email")
        timing = "within 6 hours"
        offer_action = f"Send {top_product.get('name', 'offer')}"

        mapping = {
            "switch_channel_push": ("push", "within 2 hours", offer_action),
            "switch_channel_paid": ("paid", "within 4 hours", f"Retarget {top_product.get('name', 'offer')}"),
            "backoff_24h": (channel, "backoff 24 hours", f"Reschedule {top_product.get('name', 'offer')}"),
            "suppress_7d": (channel, "suppress 7 days", "Suppress outreach — cooldown active"),
            "change_offer_category": ("email", "within 6 hours", f"Switch category offer: {top_product.get('category', 'new')}"),
            "send_default": (channel, timing, offer_action),
        }
        ch, tm, act = mapping.get(action, mapping["send_default"])
        return {"channel": ch, "timing": tm, "action_label": act}

    def _save(self) -> None:
        if self.model is None:
            return
        joblib.dump({"model": self.model, "version": self.version}, MODELS_DIR / "nba_model.joblib")

    def _load(self) -> None:
        path = MODELS_DIR / "nba_model.joblib"
        if path.exists():
            data = joblib.load(path)
            self.model = data["model"]
            self.version = data.get("version", 0)


nba_policy = NBAPolicy()
