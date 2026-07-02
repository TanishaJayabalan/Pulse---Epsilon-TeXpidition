# Pulse

A closed-loop personalization engine with LightGBM intent scoring, heuristic + ML fatigue diagnosis, sklearn NBA decision tree, FAISS similarity search, and a React marketer dashboard.

## Architecture

```text
Data Collection (events) → PeopleCloud Identity → Signal Weighting (exp decay)
    → Fatigue Heuristics → Diagnosis Classifier → Intent LightGBM
    → FAISS Top-k Filtering → NBA Policy (single-change → decision tree)
    → Confidence Routing (Logistic Regression) → LLM Reasoning (Llama 3.3)
    → HITL Marketer Review → Outcome Capture → Mini-batch Retrain (60 min)
```

## Quick Start

### 1. Generate seed data

```bash
python scripts/generate_seed_data.py
```
Creates 2,000 train + 500 test customers, behavioral events, products, and labeled outcomes in `data/`.

### 2. Backend

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8003
```
First startup trains all models and builds the FAISS index (~1-2 min on CPU).

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173

## Key Specs Implemented

| Component | Implementation |
|-----------|----------------|
| Identity | PeopleCloud mock with cookie fallback |
| Intent | LightGBM binary classifier, time-decayed features |
| Fatigue | 5 email ignores / 3 offer ignores / 5-day morning mismatch |
| Diagnosis | RandomForest multiclass (channel/timing/offer/none) |
| NBA | Single-change first, then sklearn DecisionTree combinations |
| Confidence | Logistic Regression routing to AUTO_EXECUTE, NEEDS_REVIEW, AUTO_SUPPRESSED |
| Reasoning | Groq Llama-3.3-70b-versatile generating HITL review summaries |
| Explainability | sentence-transformers + FAISS, k threshold 0.85 + collab filtering |
| Retrain | All 4 models, every 15 minutes |
| HITL | Approve / Reject / Edit with reason codes |

## API Endpoints

- `GET /api/health` — system status
- `GET /api/dashboard` — live metrics + recommendations
- `GET /api/customers` — customer list with scores
- `GET /api/customers/{id}` — profile + events + recommendation
- `GET /api/recommendations` — NBA queue
- `POST /api/recommendations/{id}/override` — marketer HITL
- `POST /api/simulate` — inject event + capture outcome
- `POST /api/retrain` — trigger mini-batch retrain
- `GET /api/models/status` — model versions

## Data Schema

```json
{
  "customer_id": "uuid",
  "interests": [{"category": "sports", "entity": "Ferrari", "weight": 0.95}],
  "favorite_driver": "Charles Leclerc",
  "favorite_brand": "Sephora",
  "fatigue_score": 0.34,
  "context": {"device": "mobile", "time_of_day": "night"}
}
```

## Local Demo Notes

- No GPU required (CPU-only LightGBM + MiniLM embeddings)
- Models persist in `models/` directory
- Dashboard auto-refreshes every 15 seconds
- Retrain scheduler runs every 60 minutes in background
