"""Configuration for Podcast Generator — powered by pydantic-settings.

Settings are loaded from environment variables with sensible defaults.
Environment variables are prefixed with PODCAST_ to avoid namespace collisions.

Key environment variables:
  PODCAST_LLM_PROVIDER     — "openai", "ollama", or "openrouter"
  PODCAST_OPENAI_API_KEY   — OpenAI API key
  PODCAST_OPENROUTER_API_KEY — OpenRouter API key
  PODCAST_TTS_PROVIDER     — "edge", "openai", or "openrouter"
  PODCAST_OUTPUT_DIR       — Directory for generated files
  PODCAST_CORS_ORIGINS     — Comma-separated list of allowed origins
"""

import os
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PODCAST_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM Configuration ---
    llm_provider: Literal["openai", "ollama", "openrouter"] = "openai"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openai/gpt-4o"

    # --- TTS Configuration ---
    tts_provider: Literal["edge", "openai", "openrouter"] = "edge"

    edge_tts_voice: str = "en-US-GuyNeural"

    openai_tts_voice: str = "alloy"
    openai_tts_model: str = "ts-1"

    openrouter_tts_model: str = "mistralai/voxtral-mini-tts-2603"
    openrouter_tts_voice: str = "en_paul_neutral"

    # --- Audio Output ---
    output_dir: str = "output"
    sample_rate: int = 44100

    # --- Beat Generation ---
    min_bpm: int = 60
    max_bpm: int = 220

    # --- CORS ---
    cors_origins: str = "*"

    # --- Job TTL ---
    job_ttl_hours: int = 24

    # --- Auth ---
    auth_token: str = ""

    # --- Persistence ---
    db_path: str = "podcast_worker.db"


# Singleton for fast import
settings = Settings()

# Convenience aliases for backward compatibility
LLM_PROVIDER = settings.llm_provider
OPENAI_API_KEY = settings.openai_api_key
OPENAI_MODEL = settings.openai_model
OLLAMA_BASE_URL = settings.ollama_base_url
OLLAMA_MODEL = settings.ollama_model
OPENROUTER_API_KEY = settings.openrouter_api_key
OPENROUTER_BASE_URL = settings.openrouter_base_url
OPENROUTER_MODEL = settings.openrouter_model
TTS_PROVIDER = settings.tts_provider
EDGE_TTS_VOICE = settings.edge_tts_voice
OPENAI_TTS_VOICE = settings.openai_tts_voice
OPENAI_TTS_MODEL = settings.openai_tts_model
OPENROUTER_TTS_MODEL = settings.openrouter_tts_model
OPENROUTER_TTS_VOICE = settings.openrouter_tts_voice
OUTPUT_DIR = settings.output_dir
SAMPLE_RATE = settings.sample_rate
MIN_BPM = settings.min_bpm
MAX_BPM = settings.max_bpm
CORS_ORIGINS = settings.cors_origins
JOB_TTL_HOURS = settings.job_ttl_hours
