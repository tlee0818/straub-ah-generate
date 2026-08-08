"""
TTS Engine - Converts podcast script text to speech audio.
Supports: edge-tts (free, no API key) and OpenAI TTS.
"""

import asyncio
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Literal
from typing import Optional

from . import config
from .config import ResolvedTTSSnapshot, RoutingConfigurationError


def _temporary_path(suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


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
@dataclass(frozen=True)
class DialogueTurn:
    role: Literal["interviewer", "guest"]
    text: str


@dataclass(frozen=True)
class TTSRequestPlan:
    plan_id: str
    strategy: str
    turns: tuple[DialogueTurn, ...]
    voice_binding: str | None = None


def parse_dialogue_turns(text: str) -> tuple[DialogueTurn, ...]:
    """Accept only canonical interviewer/SME dialogue before any remote TTS call."""
    turns: list[DialogueTurn] = []
    role: str | None = None
    lines: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^(Interviewer|SME):\s*(.+)$", line.strip())
        if match:
            if role is not None:
                content = " ".join(lines).strip()
                if not content:
                    raise RoutingConfigurationError("invalid_dialogue_turn")
                turns.append(DialogueTurn(role, content))
            role = "interviewer" if match.group(1) == "Interviewer" else "guest"
            lines = [match.group(2)]
        elif role is None or not line.strip():
            if line.strip():
                raise RoutingConfigurationError("invalid_dialogue_turn")
        else:
            lines.append(line.strip())
    if role is None:
        raise RoutingConfigurationError("invalid_dialogue_turn")
    content = " ".join(lines).strip()
    if not content:
        raise RoutingConfigurationError("invalid_dialogue_turn")
    turns.append(DialogueTurn(role, content))
    return tuple(turns)


def plan_dialogue_requests(
    text: str, snapshot: ResolvedTTSSnapshot, namespace: str = ""
) -> tuple[TTSRequestPlan, ...]:
    """Create deterministic request boundaries, uniquely scoped when persisted."""
    turns = parse_dialogue_turns(text)
    prefix = f"{namespace}:" if namespace else ""
    if snapshot.strategy == "text_to_dialogue_v3":
        plans: list[TTSRequestPlan] = []
        current: list[DialogueTurn] = []
        char_count = 0
        for turn in turns:
            if len(turn.text) > snapshot.max_scene_characters:
                raise RoutingConfigurationError("tts_turn_too_long")
            exceeds = current and (len(current) >= snapshot.max_scene_turns or char_count + len(turn.text) > snapshot.max_scene_characters)
            if exceeds:
                plans.append(TTSRequestPlan(f"{prefix}scene-{len(plans)}", snapshot.strategy, tuple(current)))
                current, char_count = [], 0
            current.append(turn)
            char_count += len(turn.text)
        if current:
            plans.append(TTSRequestPlan(f"{prefix}scene-{len(plans)}", snapshot.strategy, tuple(current)))
        return tuple(plans)
    if snapshot.strategy != "stitched_text_to_speech":
        raise RoutingConfigurationError("unsupported_tts_strategy")
    plans = []
    for turn in turns:
        for fragment in _split_dialogue_fragment(turn.text, snapshot.max_fragment_characters):
            plans.append(TTSRequestPlan(f"{prefix}fragment-{len(plans)}", snapshot.strategy, (DialogueTurn(turn.role, fragment),), snapshot.voice_bindings[turn.role]))
    return tuple(plans)


def _split_dialogue_fragment(text: str, maximum: int) -> tuple[str, ...]:
    if maximum < 1:
        raise RoutingConfigurationError("invalid_fragment_limit")
    fragments: list[str] = []
    remaining = text.strip()
    while len(remaining) > maximum:
        boundary = max(remaining.rfind(mark, 0, maximum + 1) for mark in ".!?")
        if boundary < 0:
            boundary = remaining.rfind(" ", 0, maximum + 1)
        if boundary <= 0:
            raise RoutingConfigurationError("tts_unsplittable_token")
        fragments.append(remaining[:boundary + 1].strip())
        remaining = remaining[boundary + 1:].strip()
    if remaining:
        fragments.append(remaining)
    return tuple(fragments)


def classify_tts_failure(
    status_code: int | None, dispatched: bool
) -> Literal["pre_send", "retryable", "terminal", "unknown_outcome"]:
    """Classify only outcomes that establish whether provider acceptance is possible."""
    if not dispatched:
        return "pre_send"
    if status_code in {429, 500, 503}:
        return "retryable"
    if status_code is not None:
        return "terminal"
    return "unknown_outcome"


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def synthesize_elevenlabs_plan(plan: TTSRequestPlan, snapshot: ResolvedTTSSnapshot, output_path: str) -> str:
    """Execute one planned ElevenLabs request without replaying ambiguous outcomes."""
    if snapshot.provider != "elevenlabs" or not config.settings.elevenlabs_api_key:
        raise RoutingConfigurationError("invalid_elevenlabs_configuration")
    import requests

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / ".staging" / f"{destination.name}.{plan.plan_id.replace('/', '_')}.part"
    staging.parent.mkdir(parents=True, exist_ok=True)
    # A crash after rename is recoverable locally: never send the same plan again.
    if destination.exists() and destination.stat().st_size > 0:
        return str(destination)
    if plan.strategy == "text_to_dialogue_v3":
        payload: dict[str, Any] = {
            "model_id": "eleven_v3",
            "inputs": [{"text": turn.text, "voice_id": snapshot.voice_bindings[turn.role]} for turn in plan.turns],
            "output_format": snapshot.output_format,
        }
        endpoint = "https://api.elevenlabs.io/v1/text-to-dialogue"
    else:
        turn = plan.turns[0]
        payload = {"text": turn.text, "model_id": snapshot.model_id, "voice_settings": {}}
        endpoint = f"https://api.elevenlabs.io/v1/text-to-speech/{plan.voice_binding}"
    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers={
                "xi-api-key": config.settings.elevenlabs_api_key,
                "accept": "audio/mpeg",
                "X-Request-Id": plan.plan_id,
            },
            timeout=120,
            stream=True,
        )
    except requests.RequestException as exc:
        raise RoutingConfigurationError("tts_outcome_unknown") from exc
    if not response.ok:
        raise RoutingConfigurationError(f"tts_{classify_tts_failure(response.status_code, dispatched=True)}")
    try:
        with staging.open("wb") as stream:
            for chunk in response.iter_content(65536):
                if chunk:
                    stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if staging.stat().st_size == 0:
            raise RoutingConfigurationError("tts_outcome_unknown")
        os.replace(staging, destination)
        _fsync_directory(destination.parent)
    except RoutingConfigurationError:
        raise
    except OSError as exc:
        raise RoutingConfigurationError("tts_outcome_unknown") from exc
    return str(destination)



def synthesize_edge(text: str, voice: Optional[str] = None, output_path: Optional[str] = None) -> str:
    """Synthesize speech using edge-tts (free, Microsoft Edge TTS)."""
    try:
        import edge_tts
    except ImportError:
        raise ImportError("edge-tts not installed. Run: pip install edge-tts")

    voice_name = voice or config.EDGE_TTS_VOICE
    output_path = output_path or _temporary_path(".mp3")

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
    output_path = output_path or _temporary_path(".mp3")

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
        try:
            for i, chunk in enumerate(chunks):
                chunk_path = _temporary_path(f"_chunk_{i}.mp3")
                response = client.audio.speech.create(
                    model=model_name,
                    voice=voice_name,
                    input=chunk,
                )
                response.stream_to_file(chunk_path)
                chunk_paths.append(chunk_path)
        except Exception:
            for chunk_path in chunk_paths:
                try:
                    os.remove(chunk_path)
                except OSError:
                    pass
            raise

        # Combine using pydub
        try:
            from pydub import AudioSegment
        except ImportError:
            raise ImportError("pydub not installed. Run: pip install pydub")

        combined = AudioSegment.empty()
        try:
            for cp in chunk_paths:
                seg = AudioSegment.from_mp3(cp)
                combined += seg
            combined.export(output_path, format="mp3")
        finally:
            for cp in chunk_paths:
                try:
                    os.remove(cp)
                except OSError:
                    pass

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
    output_path = output_path or _temporary_path(".mp3")

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