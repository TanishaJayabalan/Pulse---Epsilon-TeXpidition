from __future__ import annotations

from typing import Any

import faiss
import numpy as np

from ..config import MODELS_DIR, settings

_encoder = None
_index: faiss.IndexFlatIP | None = None
_product_ids: list[str] = []
_product_map: dict[str, dict] = {}
_collab_scores: dict[str, dict[str, float]] = {}


def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder


def _customer_text(profile: dict[str, Any]) -> str:
    parts = []
    for interest in profile.get("interests", []):
        parts.append(f"{interest['category']} {interest['entity']}")
    if profile.get("favorite_driver"):
        parts.append(f"driver {profile['favorite_driver']}")
    if profile.get("favorite_brand"):
        parts.append(f"brand {profile['favorite_brand']}")
    return ". ".join(parts) or "general shopper"


def _build_collaborative_scores(outcomes: list[dict]) -> None:
    global _collab_scores
    co_counts: dict[str, dict[str, int]] = {}
    for outcome in outcomes:
        if outcome.get("label") not in ("converted", "clicked"):
            continue
        cid = outcome["customer_id"]
        pid = outcome["product_id"]
        co_counts.setdefault(cid, {})
        co_counts[cid][pid] = co_counts[cid].get(pid, 0) + 1

    max_count = max((max(v.values(), default=1) for v in co_counts.values()), default=1)
    _collab_scores = {
        cid: {pid: count / max_count for pid, count in products.items()}
        for cid, products in co_counts.items()
    }


def build_index(products: list[dict], outcomes: list[dict] | None = None) -> dict[str, Any]:
    global _index, _product_ids, _product_map

    index_path = MODELS_DIR / "product_index.faiss"
    ids_path = MODELS_DIR / "product_ids.npy"

    if outcomes:
        _build_collaborative_scores(outcomes)

    if index_path.exists() and ids_path.exists() and _index is None:
        _index = faiss.read_index(str(index_path))
        _product_ids = list(np.load(str(ids_path), allow_pickle=True))
        _product_map = {p["product_id"]: p for p in products}
        return {"indexed": len(_product_ids), "dimensions": _index.d, "cached": True}

    encoder = _get_encoder()
    texts = [f"{p['name']}. {p['description']}. {p['category']} {p['entity']}" for p in products]
    embeddings = encoder.encode(texts, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    _index = faiss.IndexFlatIP(embeddings.shape[1])
    _index.add(embeddings)
    _product_ids = [p["product_id"] for p in products]
    _product_map = {p["product_id"]: p for p in products}

    faiss.write_index(_index, str(MODELS_DIR / "product_index.faiss"))
    np.save(str(MODELS_DIR / "product_ids.npy"), np.array(_product_ids))

    return {"indexed": len(products), "dimensions": embeddings.shape[1]}


def _load_index() -> None:
    global _index, _product_ids
    index_path = MODELS_DIR / "product_index.faiss"
    ids_path = MODELS_DIR / "product_ids.npy"
    if index_path.exists() and ids_path.exists() and _index is None:
        _index = faiss.read_index(str(index_path))
        _product_ids = list(np.load(str(ids_path), allow_pickle=True))


def search_top_k(
    profile: dict[str, Any],
    intent_scores: list[tuple[dict, float]] | None = None,
    k: int | None = None,
) -> list[dict[str, Any]]:
    _load_index()
    k = k or settings.top_k_products
    threshold = settings.similarity_threshold

    if _index is None or not _product_ids:
        if intent_scores:
            return [
                {**product, "similarity": round(score, 3), "combined_score": round(score, 3)}
                for product, score in intent_scores[:k]
                if score >= threshold * 0.5
            ]
        return []

    encoder = _get_encoder()
    query = encoder.encode([_customer_text(profile)], normalize_embeddings=True)
    query_vec = np.array(query, dtype=np.float32)
    scores, indices = _index.search(query_vec, min(k * 3, len(_product_ids)))

    intent_map = {p["product_id"]: s for p, s in (intent_scores or [])}
    collab = _collab_scores.get(profile.get("customer_id", ""), {})
    results = []

    for sim, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        pid = _product_ids[idx]
        product = _product_map.get(pid, {})
        intent = intent_map.get(pid, 0.3)
        collab_score = collab.get(pid, 0.0)
        combined = 0.5 * float(sim) + 0.35 * intent + 0.15 * collab_score

        if combined >= threshold or len(results) < k:
            results.append({
                **product,
                "similarity": round(float(sim), 3),
                "intent_score": round(intent, 3),
                "collab_score": round(collab_score, 3),
                "combined_score": round(combined, 3),
            })

    results.sort(key=lambda x: x["combined_score"], reverse=True)
    return results[:k]


def explain_recommendation(profile: dict[str, Any], top_products: list[dict]) -> str:
    if not top_products:
        return "Insufficient preference data; showing broad catalog recommendations."

    interest_labels = [
        f"{i['entity']} ({i['category']})" for i in profile.get("interests", [])[:3]
    ]
    product_names = ", ".join(p["name"] for p in top_products[:3])
    interests_text = ", ".join(interest_labels) if interest_labels else "your browsing history"

    return (
        f"Based on your likings for {interests_text}, these are the top items related to it: "
        f"{product_names}. Ranked by semantic similarity, intent score, and collaborative signals."
    )
