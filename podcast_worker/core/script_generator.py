"""
Script Generator - Uses an LLM to generate podcast scripts
based on the topic and target BPM energy level.
"""

import json
import os
from typing import Optional
from . import config


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


# ---------------------------------------------------------------------------
# Follow-up questions
# ---------------------------------------------------------------------------


def _get_follow_up_prompt(topic: str, script: dict) -> str:
    """Build a prompt that asks the LLM to generate follow-up questions."""
    full_text = flatten_script(script)
    title = script.get("title", "Untitled")

    return f"""You are a podcast host who just finished an episode.

EPISODE TITLE: {title}
TOPIC: {topic}
SCRIPT: {full_text}

Generate 3 thoughtful follow-up questions that a listener might ask to go deeper into this topic.
The questions should be:
- Open-ended and thought-provoking (not yes/no)
- Related to the content but exploring angles not fully covered in the script
- Written in natural, conversational language
- Varied in complexity (one beginner-friendly, one intermediate, one advanced)

Return ONLY valid JSON with this exact structure:
{{
    "follow_up_questions": [
        {{
            "question": "The full question text?",
            "context": "Brief context on why this question is relevant (1 sentence)",
            "difficulty": "beginner" | "intermediate" | "advanced"
        }}
    ]
}}
"""


def generate_follow_up_questions(
    topic: str,
    script: dict,
    provider: Optional[str] = None,
    **kwargs,
) -> dict:
    """Generate follow-up questions based on a podcast script.

    Args:
        topic: The original podcast topic.
        script: The generated script dict (with 'title' and 'segments').
        provider: LLM provider ('openai', 'ollama', or 'openrouter').
        **kwargs: Additional args passed through (api_key, model, etc.)

    Returns:
        dict with a 'follow_up_questions' list.
    """
    provider = provider or config.LLM_PROVIDER

    if provider == "openai":
        return _generate_follow_up_openai(topic, script, **kwargs)
    elif provider == "ollama":
        return _generate_follow_up_ollama(topic, script, **kwargs)
    elif provider == "openrouter":
        return _generate_follow_up_openrouter(topic, script, **kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def _generate_follow_up_openai(
    topic: str,
    script: dict,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> dict:
    """Generate follow-up questions via OpenAI."""
    from openai import OpenAI

    key = api_key or config.OPENAI_API_KEY
    if not key:
        raise ValueError("OpenAI API key not set.")

    model_name = model or config.OPENAI_MODEL
    client = OpenAI(api_key=key)
    prompt = _get_follow_up_prompt(topic, script)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You generate follow-up questions in JSON format only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.8,
    )
    return json.loads(response.choices[0].message.content)


def _generate_follow_up_ollama(
    topic: str,
    script: dict,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> dict:
    """Generate follow-up questions via Ollama."""
    import requests

    url = base_url or config.OLLAMA_BASE_URL
    model_name = model or config.OLLAMA_MODEL
    prompt = _get_follow_up_prompt(topic, script)

    resp = requests.post(
        f"{url}/api/generate",
        json={"model": model_name, "prompt": prompt, "stream": False, "format": "json"},
        timeout=120,
    )
    resp.raise_for_status()
    return json.loads(resp.json()["response"])


def _generate_follow_up_openrouter(
    topic: str,
    script: dict,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> dict:
    """Generate follow-up questions via OpenRouter."""
    from openai import OpenAI

    key = api_key or config.OPENROUTER_API_KEY
    if not key:
        raise ValueError("OpenRouter API key not set.")

    model_name = model or config.OPENROUTER_MODEL
    client = OpenAI(api_key=key, base_url=config.OPENROUTER_BASE_URL)
    prompt = _get_follow_up_prompt(topic, script)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You generate follow-up questions in JSON format only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
        extra_headers={
            "HTTP-Referer": "https://github.com/straub-ah",
            "X-Title": "BPM Podcast Generator",
        },
    )

    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    return json.loads(text.strip())


# ---------------------------------------------------------------------------
# Script summary
# ---------------------------------------------------------------------------


def _get_summary_prompt(script: dict) -> str:
    """Build a prompt that asks the LLM to summarize a podcast script."""
    full_text = flatten_script(script)
    title = script.get("title", "Untitled")

    return f"""You are given a podcast script.

EPISODE TITLE: {title}
SCRIPT: {full_text}

Generate a concise but informative summary of this podcast episode. The summary should:
- Be 2-3 paragraphs (about 150-250 words total)
- Capture the main thesis and key points
- Mention the tone/energy level implied by the content
- End with one "key takeaway" sentence
- Be written in a neutral, informative tone

Return ONLY valid JSON with this exact structure:
{{
    "title": "{title}",
    "summary": "The full summary text...",
    "key_points": ["Point 1", "Point 2", "Point 3"],
    "key_takeaway": "The single most important takeaway."
}}
"""


def generate_script_summary(
    script: dict,
    provider: Optional[str] = None,
    **kwargs,
) -> dict:
    """Generate a summary of a podcast script.

    Args:
        script: The generated script dict (with 'title' and 'segments').
        provider: LLM provider ('openai', 'ollama', or 'openrouter').
        **kwargs: Additional args passed through (api_key, model, etc.)

    Returns:
        dict with 'summary', 'key_points', and 'key_takeaway'.
    """
    provider = provider or config.LLM_PROVIDER

    if provider == "openai":
        return _generate_summary_openai(script, **kwargs)
    elif provider == "ollama":
        return _generate_summary_ollama(script, **kwargs)
    elif provider == "openrouter":
        return _generate_summary_openrouter(script, **kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def _generate_summary_openai(
    script: dict,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> dict:
    """Generate a script summary via OpenAI."""
    from openai import OpenAI

    key = api_key or config.OPENAI_API_KEY
    if not key:
        raise ValueError("OpenAI API key not set.")

    model_name = model or config.OPENAI_MODEL
    client = OpenAI(api_key=key)
    prompt = _get_summary_prompt(script)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You generate summaries in JSON format only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.5,
    )
    return json.loads(response.choices[0].message.content)


def _generate_summary_ollama(
    script: dict,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> dict:
    """Generate a script summary via Ollama."""
    import requests

    url = base_url or config.OLLAMA_BASE_URL
    model_name = model or config.OLLAMA_MODEL
    prompt = _get_summary_prompt(script)

    resp = requests.post(
        f"{url}/api/generate",
        json={"model": model_name, "prompt": prompt, "stream": False, "format": "json"},
        timeout=120,
    )
    resp.raise_for_status()
    return json.loads(resp.json()["response"])


def _generate_summary_openrouter(
    script: dict,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> dict:
    """Generate a script summary via OpenRouter."""
    from openai import OpenAI

    key = api_key or config.OPENROUTER_API_KEY
    if not key:
        raise ValueError("OpenRouter API key not set.")

    model_name = model or config.OPENROUTER_MODEL
    client = OpenAI(api_key=key, base_url=config.OPENROUTER_BASE_URL)
    prompt = _get_summary_prompt(script)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You generate summaries in JSON format only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        extra_headers={
            "HTTP-Referer": "https://github.com/straub-ah",
            "X-Title": "BPM Podcast Generator",
        },
    )

    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    return json.loads(text.strip())