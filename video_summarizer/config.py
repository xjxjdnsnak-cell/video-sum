from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="VIDEO_SUMMARIZER_",
        extra="ignore"
    )

    DB_PATH: Path = Path.home() / ".video_summarizer" / "videos.db"
    OUTPUT_DIR: Path = Path.home() / "video_summarizer_output"
    WHISPER_MODEL: str = "base"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"
    LLM_PROVIDER: Literal["openai", "ollama", "mock"] = "mock"
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama2"
    CHUNK_DURATION_MIN: int = 3
    CHUNK_DURATION_MAX: int = 5
    SRT_LANGUAGE: str = "zh"

    def ensure_directories(self):
        self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
