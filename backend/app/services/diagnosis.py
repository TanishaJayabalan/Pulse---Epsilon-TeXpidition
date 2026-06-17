from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from ..config import MODELS_DIR
from .fatigue import detect_fatigue_signals
from .signals import build_time_decayed_features

CAUSE_LABELS = ["channel_fatigue", "timing_mismatch", "offer_fatigue", "none"]


class DiagnosisModel:
    def __init__(self) -> None:
        self.model: RandomForestClassifier | None = None
        self.version = 0
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(CAUSE_LABELS)
        self._load()

    def _feature_vector(
        self,
        fatigue: dict[str, Any],
        customer: dict[str, Any],
        events: list[dict[str, Any]],
    ) -> list[float]:
        features = build_time_decayed_features(customer["customer_id"], events)
        context = customer.get("context", {})
        channel_hist = fatigue.get("channel_consecutive", {})
        return [
            fatigue.get("email_ignore_streak", 0),
            fatigue.get("max_offer_ignores", 0),
            fatigue.get("morning_ignore_days", 0),
            float(fatigue.get("channel_fatigue", False)),
            float(fatigue.get("offer_fatigue", False)),
            float(fatigue.get("timing_mismatch", False)),
            features.get("negative_7d", 0),
            features.get("email_opens_30d", 0),
            {"morning": 0, "afternoon": 1, "evening": 2, "night": 3}.get(
                context.get("time_of_day", "morning"), 0
            ),
            {"mobile": 0, "desktop": 1, "tablet": 2}.get(context.get("device", "desktop"), 1),
            channel_hist.get("email", 0),
            channel_hist.get("push", 0),
            channel_hist.get("paid", 0),
            len(customer.get("interests", [])),
        ]

    def train(self, customers: list[dict], events: list[dict], outcomes: list[dict]) -> dict[str, Any]:
        X, y = [], []
        outcome_by_customer: dict[str, list] = {}
        for outcome in outcomes:
            outcome_by_customer.setdefault(outcome["customer_id"], []).append(outcome)

        for customer in customers:
            cid = customer["customer_id"]
            cust_events = [e for e in events if e["customer_id"] == cid]
            fatigue = detect_fatigue_signals(cust_events)
            X.append(self._feature_vector(fatigue, customer, events))

            cust_outcomes = outcome_by_customer.get(cid, [])
            if cust_outcomes:
                label = cust_outcomes[-1].get("fatigue_cause", fatigue["primary_cause"])
            else:
                label = fatigue["primary_cause"]
            y.append(label)

        if len(X) < 10:
            return {"status": "skipped", "reason": "insufficient data"}

        self.model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        self.model.fit(np.array(X), y)
        self.version += 1
        self._save()
        return {"status": "trained", "version": self.version, "samples": len(X)}

    def predict(
        self,
        customer: dict[str, Any],
        events: list[dict[str, Any]],
        fatigue: dict[str, Any],
    ) -> dict[str, Any]:
        if self.model is None:
            return {
                "cause": fatigue["primary_cause"],
                "confidence": 0.6,
                "source": "heuristic",
            }

        vec = np.array([self._feature_vector(fatigue, customer, events)])
        proba = self.model.predict_proba(vec)[0]
        classes = self.model.classes_
        best_idx = int(np.argmax(proba))
        return {
            "cause": classes[best_idx],
            "confidence": round(float(proba[best_idx]), 3),
            "probabilities": {cls: round(float(p), 3) for cls, p in zip(classes, proba)},
            "source": "classifier",
        }

    def _save(self) -> None:
        if self.model is None:
            return
        path = MODELS_DIR / "diagnosis_model.joblib"
        joblib.dump({"model": self.model, "version": self.version}, path)

    def _load(self) -> None:
        path = MODELS_DIR / "diagnosis_model.joblib"
        if path.exists():
            data = joblib.load(path)
            self.model = data["model"]
            self.version = data.get("version", 0)


diagnosis_model = DiagnosisModel()
