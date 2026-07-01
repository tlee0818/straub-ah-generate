"""
Script Generator - Uses an LLM to generate podcast scripts
based on the topic and target BPM energy level.
"""

import json
import os
from typing import Optional
import config


def _get_prompt(topic: str, bpm: int, duration_minutes: int = 5) -> str:
    """Build the prompt for the LLM."""
    # Map BPM to energy level
    if bpm >= 160:
        energy = "HIGH ENERGY"
        pace = "fast-paced, punchy, exciting — like a workout hype session"
        sentence_length = "short to medium sentences, crisp delivery"
    elif bpm >= 120:
        energy = "MODERATE ENERGY"
        pace = "steady, engaging, upbeat but not rushing"
        sentence_length = "mix of short and medium sentences"
    elif bpm >= 90:
        energy = "CHILL ENERGY"
        pace = "relaxed, conversational, easygoing"
        sentence_length = "medium sentences, natural pauses"
    else:
        energy = "LOW ENERGY / DEEP FOCUS"
        pace = "calm, deliberate, thoughtful"
        sentence_length = "medium to longer sentences, reflective pauses"

    return f"""You are a podcast host creating a {duration_minutes}-minute solo educational podcast.

TOPIC: {topic}
ENERGY LEVEL: {energy}
BPM: {bpm}

The podcast will be played over a beat at {bpm} BPM, so the {pace}.

Important rules:
- {sentence_length}
- Use natural conversational language — like a knowledgeable friend explaining something
- Avoid markdown, lists, or special formatting
- Include brief pauses marked with [pause]
- Structure with: hook/intro (30s), main content (~3.5 min), outro (~30s)
- The entire script should take about {duration_minutes} minutes to read at natural pace
- Do NOT use sound effect notations like [music], [applause] — only [pause] is allowed
- NO special characters or emojis

Return ONLY valid JSON with this exact structure:
{{
    "title": "Episode title",
    "segments": [
        {{
            "segment_type": "intro",
            "text": "...",
            "approx_duration_seconds": 30
        }},
        {{
            "segment_type": "content",
            "text": "...",
            "approx_duration_seconds": 45
        }}
    ]
}}

The total of approx_duration_seconds should be about {duration_minutes * 60}.
Each content segment should be 30-60 seconds of natural speech.
"""


def generate_script_openai(
    topic: str,
    bpm: int,
    duration_minutes: int = 5,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> dict:
    """Generate a podcast script using OpenAI."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    key = api_key or config.OPENAI_API_KEY
    if not key:
        raise ValueError(
            "OpenAI API key not set. Set it in config.py or pass it via --openai-key"
        )

    model_name = model or config.OPENAI_MODEL
    client = OpenAI(api_key=key)

    prompt = _get_prompt(topic, bpm, duration_minutes)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a podcast script writer. You output raw JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.8,
    )

    raw = response.choices[0].message.content
    return json.loads(raw)


def generate_script_ollama(
    topic: str,
    bpm: int,
    duration_minutes: int = 5,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> dict:
    """Generate a podcast script using Ollama (local LLM)."""
    import requests

    url = base_url or config.OLLAMA_BASE_URL
    model_name = model or config.OLLAMA_MODEL
    prompt = _get_prompt(topic, bpm, duration_minutes)

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }

    resp = requests.post(f"{url}/api/generate", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return json.loads(data["response"])


def generate_script_openrouter(
    topic: str,
    bpm: int,
    duration_minutes: int = 5,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> dict:
    """Generate a podcast script using OpenRouter (400+ models)."""
    key = api_key or config.OPENROUTER_API_KEY
    if not key:
        raise ValueError(
            "OpenRouter API key required. Set OPENROUTER_API_KEY env var."
        )

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    model_name = model or config.OPENROUTER_MODEL
    client = OpenAI(api_key=key, base_url=config.OPENROUTER_BASE_URL)
    prompt = _get_prompt(topic, bpm, duration_minutes)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a podcast script writer. You output raw JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        extra_headers={
            "HTTP-Referer": "https://github.com/straub-ah",
            "X-Title": "BPM Podcast Generator",
        },
    )

    text = response.choices[0].message.content
    # OpenRouter may return JSON wrapped in markdown fences
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    return json.loads(text.strip())


def generate_script(
    topic: str,
    bpm: int,
    duration_minutes: int = 5,
    provider: Optional[str] = None,
    **kwargs,
) -> dict:
    """Generate a podcast script using the configured LLM provider."""
    provider = provider or config.LLM_PROVIDER

    if provider == "openai":
        return generate_script_openai(topic, bpm, duration_minutes, **kwargs)
    elif provider == "ollama":
        return generate_script_ollama(topic, bpm, duration_minutes, **kwargs)
    elif provider == "openrouter":
        return generate_script_openrouter(topic=topic, bpm=bpm, duration_minutes=duration_minutes, **kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def flatten_script(script: dict) -> str:
    """Flatten a script dict into a single text string for TTS."""
    segments = script.get("segments", [])
    return " ".join(seg["text"] for seg in segments)