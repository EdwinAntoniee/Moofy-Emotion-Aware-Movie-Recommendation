from pathlib import Path
from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = BASE_DIR / "models" / "distilbert_prod"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
DATA_DIR = BASE_DIR / "data"
ENRICHED_METADATA_FILE = DATA_DIR / "tmdb_enriched_metadata.json"
MOVIES_CSV_FILE = DATA_DIR / "tmdb_movies_with_emotions.csv"

class Settings(BaseSettings):
    PROJECT_NAME: str = "Moofy"
    VERSION: str = "1.0.3"
    API_V1_STR: str = "/api"
    SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "moofy-super-secret-cinema-jwt-key-2026")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'moofy.db'}"
    TMDB_API_KEY: str = os.getenv("TMDB_API_KEY", "")

settings = Settings()
