"""
TTS Engine - Converts podcast script text to speech audio.
Supports: edge-tts (free, no API key) and OpenAI TTS.
"""

import asyncio
import os
import tempfile
from typing import Optional

import config


def _split_into_chunks(text: str, max_chars: int = 3000) -> list:
    """Split long text into chunks for TTS APIs that have limits."""
    chunks = []
    current = ""

    for sentence in text.replace("[pause]", ". ").split(". "):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) + 2 > max_chars:
            if current:
                chunks.append(current.strip())
            current = sentence + "."
        else:
            current += sentence + ". "

    if current:
        chunks.append(current.strip())

    return chunks if chunks else [text]


def synthesize_edge(text: str, voice: Optional[str] = None, output_path: Optional[str] = None) -> str:
    """Synthesize speech using edge-tts (free, Microsoft Edge TTS)."""
    try:
        import edge_tts
    except ImportError:
        raise ImportError("edge-tts not installed. Run: pip install edge-tts")

    voice_name = voice or config.EDGE_TTS_VOICE
    output_path = output_path or tempfile.mktemp(suffix=".mp3")

    # edge-tts uses async, so we run it synchronously
    async def _run():
        communicate = edge_tts.Communicate(text=text, voice=voice_name)
        await communicate.save(output_path)

    asyncio.run(_run())
    return output_path


def synthesize_openai(
    text: str,
    api_key: Optional[str] = None,
    voice: Optional[str] = None,
    model: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """Synthesize speech using OpenAI TTS."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai not installed. Run: pip install openai")

    key = api_key or config.OPENAI_API_KEY
    if not key:
        raise ValueError("OpenAI API key not set.")

    voice_name = voice or config.OPENAI_TTS_VOICE
    model_name = model or config.OPENAI_TTS_MODEL
    output_path = output_path or tempfile.mktemp(suffix=".mp3")

    client = OpenAI(api_key=key)

    # OpenAI TTS has a 4096 character limit per request
    chunks = _split_into_chunks(text, max_chars=4000)

    if len(chunks) == 1:
        response = client.audio.speech.create(
            model=model_name,
            voice=voice_name,
            input=text,
        )
        response.stream_to_file(output_path)
    else:
        # Multiple chunks — synthesize each and combine
        chunk_paths = []
        for i, chunk in enumerate(chunks):
            chunk_path = tempfile.mktemp(suffix=f"_chunk_{i}.mp3")
            response = client.audio.speech.create(
                model=model_name,
                voice=voice_name,
                input=chunk,
            )
            response.stream_to_file(chunk_path)
            chunk_paths.append(chunk_path)

        # Combine using pydub
        try:
            from pydub import AudioSegment
        except ImportError:
            raise ImportError("pydub not installed. Run: pip install pydub")

        combined = AudioSegment.empty()
        for cp in chunk_paths:
            seg = AudioSegment.from_mp3(cp)
            combined += seg
            os.remove(cp)

        combined.export(output_path, format="mp3")

    return output_path


def synthesize_openrouter(
    text: str,
    output_path: Optional[str] = None,
    voice: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """Synthesize speech using OpenRouter TTS (9+ models through one API)."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai not installed. Run: pip install openai")

    key = api_key or config.OPENROUTER_API_KEY
    if not key:
        raise ValueError(
            "OpenRouter API key required. Set OPENROUTER_API_KEY env var."
        )

    voice_name = voice or config.OPENROUTER_TTS_VOICE
    output_path = output_path or tempfile.mktemp(suffix=".mp3")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    client = OpenAI(api_key=key, base_url=config.OPENROUTER_BASE_URL)
    response = client.audio.speech.create(
        model=config.OPENROUTER_TTS_MODEL,
        voice=voice_name,
        input=text,
        response_format="mp3",
        extra_headers={
            "HTTP-Referer": "https://github.com/straub-ah",
            "X-Title": "BPM Podcast Generator",
        },
    )
    response.stream_to_file(output_path)
    return output_path


def synthesize(
    text: str,
    provider: Optional[str] = None,
    output_path: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Synthesize text to speech using the configured provider.

    Returns the path to the generated audio file.
    """
    provider = provider or config.TTS_PROVIDER

    if provider == "edge":
        return synthesize_edge(text, output_path=output_path, **kwargs)
    elif provider == "openai":
        return synthesize_openai(text, output_path=output_path, **kwargs)
    elif provider == "openrouter":
        return synthesize_openrouter(text, output_path=output_path, **kwargs)
    else:
        raise ValueError(f"Unknown TTS provider: {provider}")


def convert_to_wav(input_path: str, output_path: Optional[str] = None) -> str:
    """Convert an MP3/WAV to a standardized WAV file using pydub."""
    from pydub import AudioSegment

    output_path = output_path or input_path.replace(".mp3", ".wav").replace(".ogg", ".wav")

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(config.SAMPLE_RATE).set_channels(1)
    audio.export(output_path, format="wav")
    return output_path