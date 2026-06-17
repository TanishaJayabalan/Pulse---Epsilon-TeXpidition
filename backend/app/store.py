from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import DATA_DIR


class DataStore:
    def __init__(self) -> None:
        self.customers: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.products: dict[str, dict[str, Any]] = {}
        self.outcomes: list[dict[str, Any]] = []
        self.recommendations: dict[str, dict[str, Any]] = {}
        self.overrides: list[dict[str, Any]] = []
        self.action_history: dict[str, list[dict[str, Any]]] = {}
        self.metrics_log: list[dict[str, Any]] = []
        self.model_status: dict[str, Any] = {
            "intent_model": {"version": 0, "last_trained": None, "samples": 0},
            "diagnosis_model": {"version": 0, "last_trained": None, "samples": 0},
            "nba_model": {"version": 0, "last_trained": None, "samples": 0},
            "retrain_count": 0,
            "last_retrain": None,
        }

    def load_seed(self) -> None:
        if self.customers:
            return
        self._load_json_file("customers_train.json", self._ingest_customers)
        self._load_json_file("events_train.json", self._ingest_events)
        self._load_json_file("products.json", self._ingest_products)
        self._load_json_file("outcomes_train.json", self._ingest_outcomes)

    def _load_json_file(self, filename: str, ingest_fn) -> None:
        path = DATA_DIR / filename
        if not path.exists():
            return
        with path.open(encoding="utf-8") as handle:
            ingest_fn(json.load(handle))

    def _ingest_customers(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self.customers[row["customer_id"]] = row

    def _ingest_events(self, rows: list[dict[str, Any]]) -> None:
        self.events.extend(rows)

    def _ingest_products(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self.products[row["product_id"]] = row

    def _ingest_outcomes(self, rows: list[dict[str, Any]]) -> None:
        self.outcomes.extend(rows)

    def list_customers(self) -> list[dict[str, Any]]:
        return list(self.customers.values())

    def get_customer(self, customer_id: str) -> dict[str, Any] | None:
        return self.customers.get(customer_id)

    def list_events(self, customer_id: str | None = None) -> list[dict[str, Any]]:
        if customer_id:
            return [event for event in self.events if event["customer_id"] == customer_id]
        return list(self.events)

    def add_event(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def list_products(self) -> list[dict[str, Any]]:
        return list(self.products.values())

    def list_outcomes(self) -> list[dict[str, Any]]:
        return list(self.outcomes)

    def add_outcome(self, outcome: dict[str, Any]) -> None:
        self.outcomes.append(outcome)

    def upsert_recommendation(self, rec: dict[str, Any]) -> None:
        self.recommendations[rec["id"]] = rec

    def get_recommendation(self, rec_id: str) -> dict[str, Any] | None:
        return self.recommendations.get(rec_id)

    def list_recommendations(self) -> list[dict[str, Any]]:
        return list(self.recommendations.values())

    def add_override(self, override: dict[str, Any]) -> None:
        self.overrides.append(override)

    def list_overrides(self) -> list[dict[str, Any]]:
        return list(self.overrides)

    def add_action_history(self, customer_id: str, entry: dict[str, Any]) -> None:
        self.action_history.setdefault(customer_id, []).append(entry)

    def get_action_history(self, customer_id: str) -> list[dict[str, Any]]:
        return self.action_history.get(customer_id, [])

    def log_metrics(self, snapshot: dict[str, Any]) -> None:
        self.metrics_log.append(snapshot)
        if len(self.metrics_log) > 500:
            self.metrics_log = self.metrics_log[-500:]

    def reset(self) -> None:
        self.__init__()
        self.load_seed()

    def save_runtime_snapshot(self) -> None:
        runtime_dir = DATA_DIR / "runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "events": self.events[-2000:],
            "outcomes": self.outcomes[-2000:],
            "overrides": self.overrides[-500:],
            "model_status": self.model_status,
        }
        with (runtime_dir / "snapshot.json").open("w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, indent=2)


store = DataStore()
