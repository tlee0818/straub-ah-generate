# Configuration for Podcast Generator

# --- LLM Configuration ---
# Choose your LLM provider: "openai", "ollama", or "openrouter"
LLM_PROVIDER = "openai"

# OpenAI settings (used when LLM_PROVIDER = "openai" or for OpenAI TTS)
OPENAI_API_KEY = ""
OPENAI_MODEL = "gpt-4o"  # or gpt-4o-mini for cheaper

# Ollama settings (used when LLM_PROVIDER = "ollama")
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"

# OpenRouter settings (used when LLM_PROVIDER = "openrouter" or TTS_PROVIDER = "openrouter")
# NOTE: Set your API key via the OPENROUTER_API_KEY environment variable, not here!
import os
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = "openai/gpt-4o"  # Any model from openrouter.ai/models

# --- TTS Configuration ---
# Choose: "edge" (free, no key), "openai" (premium, needs key), "openrouter"
TTS_PROVIDER = "edge"

# Edge TTS voice
EDGE_TTS_VOICE = "en-US-GuyNeural"  # Male voice
# EDGE_TTS_VOICE = "en-US-JennyNeural"  # Female voice

# OpenAI TTS voice (used when TTS_PROVIDER = "openai")
OPENAI_TTS_VOICE = "alloy"
OPENAI_TTS_MODEL = "tts-1"

# OpenRouter TTS (access to 9+ TTS models through one API)
OPENROUTER_TTS_MODEL = "mistralai/voxtral-mini-tts-2603"
OPENROUTER_TTS_VOICE = "en_paul_neutral"

# --- Audio Output ---
OUTPUT_DIR = "output"
SAMPLE_RATE = 44100

# --- Beat Generation ---
# BPM range validation
MIN_BPM = 60
MAX_BPM = 220
