import os
from pathlib import Path


class Settings:
    REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
    VLLM_URL = os.environ.get("VLLM_URL", "http://vllm-whisper:8000")
    PYANNOTE_URL = os.environ.get("PYANNOTE_URL", "http://pyannote:8000")
    RESEMBLYZER_URL = os.environ.get("RESEMBLYZER_URL", "http://resemblyzer:8000")
    WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "openai/whisper-large-v3")
    DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
    DB_PATH = Path(os.environ.get("DB_PATH", "/data/meeting_agent.db"))


settings = Settings()
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
