from pathlib import Path

from dotenv import dotenv_values
from pydantic import BaseModel, Field


class Settings(BaseModel):
    openai_api_key: str = Field(min_length=1)
    ai_provider: str = Field(default="openai", min_length=1)
    ai_model: str = Field(default="gpt-4.1-mini", min_length=1)


def load_settings(env_file: str | Path = ".env") -> Settings:
    values = dotenv_values(env_file)
    return Settings(
        openai_api_key=values.get("OPENAI_API_KEY", ""),
        ai_provider=values.get("AI_PROVIDER", "openai"),
        ai_model=values.get("AI_MODEL", "gpt-4.1-mini"),
    )

