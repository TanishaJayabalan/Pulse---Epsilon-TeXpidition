from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from ..config import MODELS_DIR

CONFIDENCE_MODEL_PATH = MODELS_DIR / "confidence_model.joblib"

FATIGUE_TYPE_MAP = {
    "none": 0,
    "channel_fatigue": 1,
    "channel": 1,
    "timing_mismatch": 2,
    "timing": 2,
    "offer_fatigue": 3,
    "offer": 3,
    "unsubscribe": 3,
}

CHANNEL_MAP = {
    "email": 0,
    "push": 1,
    "paid": 2,
    "onsite": 3,
}

LABEL_TO_ZONE = {
    0: ("AUTO_SUPPRESSED", "red"),
    1: ("NEEDS_REVIEW", "amber"),
    2: ("AUTO_EXECUTE", "green"),
}

# LightGBM conversion probabilities cluster low (~0.05–0.45); stretch before rule checks.
INTENT_CALIBRATION_FLOOR = 0.05
INTENT_CALIBRATION_SPAN = 0.40
MIN_CATEGORY_REVIEWS = 1
MIN_HITL_REVIEWS = 1

# Calibrated intent thresholds (0–1 scale after stretching).
LIVE_SUPPRESS_INTENT = 0.12
LIVE_EXECUTE_INTENT = 0.78
LIVE_EXECUTE_FATIGUE = 0.35
LIVE_FATIGUE_SUPPRESS = 0.72
LIVE_FATIGUE_SUPPRESS_INTENT = 0.45


class ConfidenceModel:
    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=500)
        self.version = 1
        self.trained = False
        self.feedback_buffer: list[dict[str, Any]] = []

    def _encode_fatigue_type(self, fatigue_type: str) -> int:
        return FATIGUE_TYPE_MAP.get(str(fatigue_type).lower(), 0)

    def _encode_channel(self, channel: str) -> int:
        return CHANNEL_MAP.get(str(channel).lower(), 0)

    def _feature_vector(
        self,
        intent_score: float,
        fatigue_score: float,
        fatigue_type: str,
        channel: str,
        days_since_last_conversion: float,
        hitl_override_rate: float,
        marketer_reject_rate: float,
    ) -> list[float]:
        return [
            float(np.clip(intent_score, 0.0, 1.0)),
            float(np.clip(fatigue_score, 0.0, 1.0)),
            float(self._encode_fatigue_type(fatigue_type)),
            float(self._encode_channel(channel)),
            float(min(max(days_since_last_conversion, 0.0), 30.0)),
            float(np.clip(hitl_override_rate, 0.0, 1.0)),
            float(np.clip(marketer_reject_rate, 0.0, 1.0)),
        ]

    def _normalize_score(self, value: float) -> float:
        if value > 1.0:
            value = value / 100.0
        return float(np.clip(value, 0.0, 1.0))

    def _calibrate_intent(self, intent_score: float) -> float:
        """Stretch typical model output (~0.08–0.55) into 0–1 decision space."""
        intent_score = self._normalize_score(intent_score)
        calibrated = (intent_score - INTENT_CALIBRATION_FLOOR) / INTENT_CALIBRATION_SPAN
        return float(np.clip(calibrated, 0.0, 1.0))

    def _assign_bootstrap_label(
        self,
        intent_score: float,
        fatigue_score: float,
        fatigue_type: str,
        hitl_override_rate: float,
        marketer_reject_rate: float,
    ) -> int:
        if (
            intent_score >= 0.85
            and fatigue_score <= 0.3
            and hitl_override_rate <= 0.2
        ):
            return 2
        if (
            intent_score <= 0.40
            or str(fatigue_type).lower() == "unsubscribe"
            or marketer_reject_rate >= 0.7
        ):
            return 0
        return 1

    def _assign_live_label(
        self,
        intent_score: float,
        fatigue_score: float,
        fatigue_type: str,
        hitl_override_rate: float,
        marketer_reject_rate: float,
        *,
        nba_suppressed: bool = False,
    ) -> int:
        """Apply spec rules on calibrated scores for real pipeline data."""
        intent = self._calibrate_intent(intent_score)
        fatigue = self._normalize_score(fatigue_score)

        if nba_suppressed or str(fatigue_type).lower() == "unsubscribe":
            return 0
        if marketer_reject_rate >= 0.7:
            return 0
        if (
            intent >= LIVE_EXECUTE_INTENT
            and fatigue <= LIVE_EXECUTE_FATIGUE
            and hitl_override_rate <= 0.25
        ):
            return 2
        if intent <= LIVE_SUPPRESS_INTENT:
            return 0
        if fatigue >= LIVE_FATIGUE_SUPPRESS and intent <= LIVE_FATIGUE_SUPPRESS_INTENT:
            return 0
        return 1

    def _rule_confidence(
        self,
        rule_label: int,
        intent_score: float,
        fatigue_score: float,
        *,
        nba_suppressed: bool = False,
    ) -> float:
        """Human-readable confidence in the zone decision (not sklearn class probability)."""
        intent = self._calibrate_intent(intent_score)
        fatigue = self._normalize_score(fatigue_score)

        if rule_label == 0:
            signals: list[float] = []
            if intent <= LIVE_SUPPRESS_INTENT:
                signals.append(1.0 - (intent / LIVE_SUPPRESS_INTENT))
            if fatigue >= LIVE_FATIGUE_SUPPRESS:
                signals.append((fatigue - LIVE_FATIGUE_SUPPRESS) / (1.0 - LIVE_FATIGUE_SUPPRESS))
            if nba_suppressed:
                signals.append(0.88)
            return float(np.clip(max(signals) if signals else 0.65, 0.35, 0.95))

        if rule_label == 2:
            execute_strength = min(
                intent / LIVE_EXECUTE_INTENT,
                (LIVE_EXECUTE_FATIGUE - fatigue) / LIVE_EXECUTE_FATIGUE if fatigue < LIVE_EXECUTE_FATIGUE else 0.5,
            )
            return float(np.clip(0.55 + execute_strength * 0.40, 0.55, 0.95))

        # NEEDS_REVIEW — clearest when signals sit in the middle band
        middle_strength = 1.0 - abs(intent - 0.625) / 0.625
        return float(np.clip(0.48 + middle_strength * 0.35, 0.40, 0.78))

    def _generate_bootstrap_data(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(42)
        rows: list[list[float]] = []
        labels: list[int] = []
        per_class = 100

        for target_label in (0, 1, 2):
            generated = 0
            attempts = 0
            while generated < per_class and attempts < 5000:
                attempts += 1
                if target_label == 2:
                    intent_score = float(rng.uniform(0.85, 1.0))
                    fatigue_score = float(rng.uniform(0.0, 0.3))
                    hitl_override_rate = float(rng.uniform(0.0, 0.2))
                    marketer_reject_rate = float(rng.uniform(0.0, 0.4))
                    fatigue_type = rng.choice(["none", "channel", "timing"])
                elif target_label == 0:
                    scenario = int(rng.integers(0, 3))
                    marketer_reject_rate = float(rng.uniform(0.0, 0.69))
                    if scenario == 0:
                        intent_score = float(rng.uniform(0.0, 0.40))
                        fatigue_score = float(rng.uniform(0.0, 1.0))
                        fatigue_type = rng.choice(["none", "channel", "timing", "offer"])
                    elif scenario == 1:
                        intent_score = float(rng.uniform(0.0, 1.0))
                        fatigue_score = float(rng.uniform(0.0, 1.0))
                        fatigue_type = "unsubscribe"
                    else:
                        intent_score = float(rng.uniform(0.0, 1.0))
                        fatigue_score = float(rng.uniform(0.0, 1.0))
                        fatigue_type = rng.choice(["none", "channel", "timing", "offer"])
                        marketer_reject_rate = float(rng.uniform(0.7, 1.0))
                    hitl_override_rate = float(rng.uniform(0.0, 1.0))
                else:
                    intent_score = float(rng.uniform(0.41, 0.84))
                    fatigue_score = float(rng.uniform(0.31, 0.9))
                    hitl_override_rate = float(rng.uniform(0.21, 1.0))
                    marketer_reject_rate = float(rng.uniform(0.0, 0.69))
                    fatigue_type = rng.choice(["none", "channel", "timing", "offer"])

                channel = rng.choice(list(CHANNEL_MAP.keys()))
                days_since_last_conversion = float(rng.uniform(0.0, 30.0))

                noise = rng.normal(0.0, 0.05, size=5)
                intent_score = float(np.clip(intent_score + noise[0], 0.0, 1.0))
                fatigue_score = float(np.clip(fatigue_score + noise[1], 0.0, 1.0))
                days_since_last_conversion = float(
                    np.clip(days_since_last_conversion + noise[2], 0.0, 30.0)
                )
                hitl_override_rate = float(np.clip(hitl_override_rate + noise[3], 0.0, 1.0))
                marketer_reject_rate = float(np.clip(marketer_reject_rate + noise[4], 0.0, 1.0))

                label = self._assign_bootstrap_label(
                    intent_score,
                    fatigue_score,
                    fatigue_type,
                    hitl_override_rate,
                    marketer_reject_rate,
                )
                if label != target_label:
                    continue

                rows.append(
                    self._feature_vector(
                        intent_score,
                        fatigue_score,
                        fatigue_type,
                        channel,
                        days_since_last_conversion,
                        hitl_override_rate,
                        marketer_reject_rate,
                    )
                )
                labels.append(label)
                generated += 1

        return np.array(rows, dtype=float), np.array(labels, dtype=int)

    def _bootstrap_and_train(self) -> None:
        try:
            bootstrap_X, bootstrap_y = self._generate_bootstrap_data()
            self.model.fit(bootstrap_X, bootstrap_y)
            self.trained = True
        except Exception:
            self.trained = False

    def _compute_hitl_override_rate(self, customer_id: str, store: Any) -> float:
        try:
            history = store.get_action_history(customer_id)
            hitl_entries = [
                entry
                for entry in history
                if entry.get("status") in ("approved", "rejected", "edited")
            ]
            if not hitl_entries or len(hitl_entries) < MIN_HITL_REVIEWS:
                return 0.0
            overrides = sum(
                1
                for entry in hitl_entries
                if entry.get("status") in ("rejected", "edited")
            )
            return overrides / len(hitl_entries)
        except Exception:
            return 0.0

    def _compute_marketer_reject_rate(self, offer_category: str, store: Any) -> float:
        try:
            for history in store.action_history.values():
                for entry in history:
                    if entry.get("offer_category") == offer_category and entry.get("status") == "rejected" and entry.get("reason") == "stock_unavailable":
                        return 1.0
            return 0.0
        except Exception:
            return 0.0

    def predict(
        self,
        intent_score: float,
        fatigue_score: float,
        fatigue_type: str,
        channel: str,
        days_since_last_conversion: float,
        customer_id: str,
        offer_category: str,
        store: Any,
        *,
        nba_suppressed: bool = False,
    ) -> dict[str, Any]:
        try:
            if not self.trained:
                self._bootstrap_and_train()

            intent_norm = self._normalize_score(intent_score)
            fatigue_norm = self._normalize_score(fatigue_score)

            hitl_override_rate = self._compute_hitl_override_rate(customer_id, store)
            marketer_reject_rate = self._compute_marketer_reject_rate(offer_category, store)

            features = np.array(
                [
                    self._feature_vector(
                        intent_norm,
                        fatigue_norm,
                        fatigue_type,
                        channel,
                        days_since_last_conversion,
                        hitl_override_rate,
                        marketer_reject_rate,
                    )
                ],
                dtype=float,
            )
            self.model.predict_proba(features)
            rule_label = self._assign_live_label(
                intent_score,
                fatigue_score,
                fatigue_type,
                hitl_override_rate,
                marketer_reject_rate,
                nba_suppressed=nba_suppressed,
            )
            zone, color = LABEL_TO_ZONE.get(rule_label, ("NEEDS_REVIEW", "amber"))
            confidence = self._rule_confidence(
                rule_label,
                intent_score,
                fatigue_score,
                nba_suppressed=nba_suppressed,
            )
            
            suppression_reason = None
            if rule_label == 0:
                reasons = []
                if nba_suppressed:
                    reasons.append("NBA cooldown policy")
                if str(fatigue_type).lower() == "unsubscribe":
                    reasons.append("user unsubscribed")
                if marketer_reject_rate >= 0.7:
                    reasons.append("stock unavailable")
                if self._calibrate_intent(intent_score) <= LIVE_SUPPRESS_INTENT and not nba_suppressed:
                    reasons.append("weak intent")
                if self._normalize_score(fatigue_score) >= LIVE_FATIGUE_SUPPRESS:
                    reasons.append("elevated fatigue")
                suppression_reason = " and ".join(reasons) if reasons else "high model rejection risk"
                
            return {
                "zone": zone,
                "confidence": confidence,
                "color": color,
                "suppression_reason": suppression_reason,
            }
        except Exception:
            return {
                "zone": "NEEDS_REVIEW",
                "confidence": 0.5,
                "color": "amber",
            }

    def record_feedback(
        self,
        intent_score: float,
        fatigue_score: float,
        fatigue_type: str,
        channel: str,
        days_since_last_conversion: float,
        hitl_override_rate: float,
        marketer_reject_rate: float,
        label: int,
    ) -> None:
        try:
            if intent_score > 1.0:
                intent_score = intent_score / 100.0
            if fatigue_score > 1.0:
                fatigue_score = fatigue_score / 100.0

            self.feedback_buffer.append(
                {
                    "features": self._feature_vector(
                        intent_score,
                        fatigue_score,
                        fatigue_type,
                        channel,
                        days_since_last_conversion,
                        hitl_override_rate,
                        marketer_reject_rate,
                    ),
                    "label": int(label),
                }
            )
            if len(self.feedback_buffer) >= 1:
                bootstrap_X, bootstrap_y = self._generate_bootstrap_data()
                self.retrain(bootstrap_X, bootstrap_y)
        except Exception:
            return

    def retrain(self, bootstrap_X: np.ndarray, bootstrap_y: np.ndarray) -> None:
        try:
            feedback_X = np.array(
                [entry["features"] for entry in self.feedback_buffer],
                dtype=float,
            )
            feedback_y = np.array(
                [entry["label"] for entry in self.feedback_buffer],
                dtype=int,
            )
            combined_X = np.vstack([bootstrap_X, feedback_X])
            combined_y = np.concatenate([bootstrap_y, feedback_y])
            self.model.fit(combined_X, combined_y)
            self.trained = True
            self.version += 1
            self.feedback_buffer = []
            self.save(CONFIDENCE_MODEL_PATH)
        except Exception:
            return

    def save(self, path: Any) -> None:
        try:
            joblib.dump(self, path)
        except Exception:
            return

    @staticmethod
    def load(path: Any) -> ConfidenceModel:
        return joblib.load(path)


confidence_model = ConfidenceModel()
