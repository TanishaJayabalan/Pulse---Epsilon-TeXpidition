from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    frontend_origin: str = "http://localhost:5173"
    retrain_interval_minutes: int = 60
    similarity_threshold: float = 0.85
    top_k_products: int = 5
    decay_half_life_hours: float = 48.0


settings = Settings()
MODELS_DIR.mkdir(parents=True, exist_ok=True)
