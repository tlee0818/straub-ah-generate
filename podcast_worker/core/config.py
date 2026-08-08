"""Server-owned worker configuration and immutable generation routing snapshots."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

from pydantic_settings import BaseSettings, SettingsConfigDict

LLM_PURPOSES = (
    "outline",
    "research_brief",
    "subtopic_research",
    "dialogue_draft",
    "fact_verification",
    "follow_up",
    "summary",
)
V1_LLM_PURPOSES = LLM_PURPOSES[:5]


class RoutingConfigurationError(ValueError):
    """Raised before provider work when an operator routing configuration is invalid."""


@dataclass(frozen=True)
class LLMRoute:
    purpose: str
    provider: Literal["openai", "ollama", "openrouter"]
    model: str
    dialect: Literal["openai_json_object", "ollama_format_json", "openrouter_legacy_prompt_strict_parse"]
    temperature: float = 0.8
    timeout_seconds: int = 120


@dataclass(frozen=True)
class ResolvedExecutionSnapshot:
    profile_id: str
    revision: str
    routes: Mapping[str, LLMRoute]

    def route_for(self, purpose: str) -> LLMRoute:
        try:
            return self.routes[purpose]
        except KeyError as exc:
            raise RoutingConfigurationError(f"unsupported_llm_purpose:{purpose}") from exc

    def canonical_json(self) -> str:
        payload = {
            "profile_id": self.profile_id,
            "routes": {name: asdict(self.routes[name]) for name in sorted(self.routes)},
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class ResolvedTTSSnapshot:
    profile_id: str
    revision: str
    provider: str
    strategy: str
    model_id: str | None
    output_format: str
    max_scene_characters: int
    max_scene_turns: int
    max_fragment_characters: int
    voice_bindings: Mapping[str, str]
    max_attempts: int
    context_max_request_ids: int = 0
    context_max_age_seconds: int = 0
    continuity_after_expiry: str | None = None
    max_concurrent_requests: int = 1
def execution_snapshot_from_payload(payload: Mapping[str, Any], revision: str | None = None) -> ResolvedExecutionSnapshot:
    """Hydrate the persisted server-only LLM snapshot without consulting Settings."""
    profile_id, routes_payload = payload.get("profile_id"), payload.get("routes")
    if not isinstance(profile_id, str) or not isinstance(routes_payload, Mapping):
        raise RoutingConfigurationError("invalid_persisted_llm_snapshot")
    routes: dict[str, LLMRoute] = {}
    for purpose in V1_LLM_PURPOSES:
        raw = routes_payload.get(purpose)
        if not isinstance(raw, Mapping):
            raise RoutingConfigurationError("invalid_persisted_llm_snapshot")
        try:
            routes[purpose] = LLMRoute(purpose, raw["provider"], raw["model"], raw["dialect"],
                                       float(raw.get("temperature", 0.8)), int(raw.get("timeout_seconds", 120)))
        except (KeyError, TypeError, ValueError) as exc:
            raise RoutingConfigurationError("invalid_persisted_llm_snapshot") from exc
    resolved_revision = revision or payload.get("revision")
    if not isinstance(resolved_revision, str) or not resolved_revision:
        raise RoutingConfigurationError("invalid_persisted_llm_snapshot")
    return ResolvedExecutionSnapshot(profile_id, resolved_revision, MappingProxyType(routes))


def tts_snapshot_from_payload(payload: Mapping[str, Any], revision: str | None = None) -> ResolvedTTSSnapshot:
    """Hydrate raw server-only bindings from durable storage, never public input."""
    bindings = payload.get("voice_bindings")
    required = ("profile_id", "provider", "strategy", "output_format", "max_attempts")
    if any(key not in payload for key in required) or not isinstance(bindings, Mapping):
        raise RoutingConfigurationError("invalid_persisted_tts_snapshot")
    if not all(isinstance(bindings.get(role), str) and bindings[role] for role in ("interviewer", "guest")):
        raise RoutingConfigurationError("invalid_persisted_tts_snapshot")
    resolved_revision = revision or payload.get("revision")
    if not isinstance(resolved_revision, str) or not resolved_revision:
        raise RoutingConfigurationError("invalid_persisted_tts_snapshot")
    return ResolvedTTSSnapshot(
        str(payload["profile_id"]), resolved_revision, str(payload["provider"]), str(payload["strategy"]),
        payload.get("model_id"), str(payload["output_format"]), int(payload.get("max_scene_characters", 0)),
        int(payload.get("max_scene_turns", 0)), int(payload.get("max_fragment_characters", 0)),
        MappingProxyType(dict(bindings)), int(payload["max_attempts"]),
        int(payload.get("context_max_request_ids", 0)), int(payload.get("context_max_age_seconds", 0)),
        payload.get("continuity_after_expiry"), int(payload.get("max_concurrent_requests", 1)),
    )



class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PODCAST_", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Literal["openai", "ollama", "openrouter"] = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openai/gpt-4o"
    llm_routing_profiles_json: str = ""

    tts_provider: Literal["edge", "openai", "openrouter", "elevenlabs"] = "edge"
    edge_tts_voice: str = "en-US-GuyNeural"
    openai_tts_voice: str = "alloy"
    openai_tts_model: str = "tts-1"
    openrouter_tts_model: str = "mistralai/voxtral-mini-tts-2603"
    openrouter_tts_voice: str = "en_paul_neutral"
    elevenlabs_api_key: str = ""
    tts_profiles_json: str = ""

    output_dir: str = "output"
    sample_rate: int = 44100
    max_text_chars: int = 20_000
    max_upload_bytes: int = 25 * 1024 * 1024
    max_concurrent_generations: int = 2
    realism_level: Literal["subtle", "natural", "expressive"] = "natural"
    allow_nonverbal_cues: bool = True
    min_bpm: int = 60
    max_bpm: int = 220
    cors_origins: str = "*"
    job_ttl_hours: int = 24
    auth_token: str = ""
    allow_insecure_dev_auth: bool = False
    db_path: str = "podcast_worker.db"


def _json_object(value: str, name: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise RoutingConfigurationError(f"invalid_{name}") from exc
    if not isinstance(decoded, dict):
        raise RoutingConfigurationError(f"invalid_{name}")
    return decoded


_BUNDLED_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _profile_document(env_value: str, filename: str, name: str) -> dict[str, Any]:
    if env_value.strip():
        return _json_object(env_value, name)
    try:
        decoded = json.loads((_BUNDLED_CONFIG_DIR / filename).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RoutingConfigurationError(f"invalid_bundled_{name}") from exc
    if not isinstance(decoded, dict):
        raise RoutingConfigurationError(f"invalid_bundled_{name}")
    return decoded


def llm_profile_document() -> dict[str, Any]:
    return _profile_document(settings.llm_routing_profiles_json, "llm_profiles.json", "llm_routing_profiles_json")


def tts_profile_document() -> dict[str, Any]:
    return _profile_document(settings.tts_profiles_json, "tts_profiles.json", "tts_profiles_json")


def _revision(prefix: str, value: Mapping[str, Any]) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"{prefix}_{hashlib.sha256(canonical.encode()).hexdigest()[:20]}"


def _scalar_llm_route(purpose: str) -> LLMRoute:
    provider = settings.llm_provider
    if provider == "openai":
        return LLMRoute(purpose, provider, settings.openai_model, "openai_json_object")
    if provider == "ollama":
        return LLMRoute(purpose, provider, settings.ollama_model, "ollama_format_json")
    return LLMRoute(purpose, provider, settings.openrouter_model, "openrouter_legacy_prompt_strict_parse")


def resolve_llm_profile(profile_id: str | None = None) -> ResolvedExecutionSnapshot:
    """Resolve an immutable internal LLM snapshot; callers persist its JSON/revision."""
    source = llm_profile_document()
    allowed = {"schema_version", "default_profile_id", "profiles"}
    if set(source) != allowed or not isinstance(source["profiles"], list):
        raise RoutingConfigurationError("invalid_llm_routing_profiles_json")
    selected = source["default_profile_id"] if profile_id in {None, "default"} else profile_id
    profile = next((item for item in source["profiles"] if isinstance(item, dict) and item.get("id") == selected), None)
    if profile is None:
        raise RoutingConfigurationError("unknown_llm_profile")
    routes_source = profile.get("routes")
    if not isinstance(routes_source, dict) or set(routes_source) != set(V1_LLM_PURPOSES):
        raise RoutingConfigurationError("invalid_llm_profile_routes")
    routes: dict[str, LLMRoute] = {}
    for purpose, raw in routes_source.items():
        if not isinstance(raw, dict):
            raise RoutingConfigurationError("invalid_llm_route")
        provider, model, dialect = raw.get("provider"), raw.get("model"), raw.get("dialect")
        if provider not in {"openai", "ollama", "openrouter"} or not isinstance(model, str) or not model:
            raise RoutingConfigurationError("invalid_llm_route")
        expected = {"openai": "openai_json_object", "ollama": "ollama_format_json", "openrouter": "openrouter_legacy_prompt_strict_parse"}[provider]
        if dialect != expected:
            raise RoutingConfigurationError("invalid_llm_route_dialect")
        routes[purpose] = LLMRoute(purpose, provider, model, dialect, float(raw.get("temperature", 0.8)), int(raw.get("timeout_seconds", 120)))
    # Legacy endpoints intentionally retain scalar server-only routing.
    routes.update({purpose: _scalar_llm_route(purpose) for purpose in LLM_PURPOSES[5:]})
    payload = {name: asdict(route) for name, route in routes.items()}
    return ResolvedExecutionSnapshot(str(selected), _revision("rte", payload), MappingProxyType(routes))


def resolve_tts_profile(profile_id: str | None = None, interviewer_voice_id: str | None = None, guest_voice_id: str | None = None) -> ResolvedTTSSnapshot:
    """Resolve server-only raw voice bindings. This object must never be serialized to public DTOs."""
    source = tts_profile_document()
    if set(source) != {"schema_version", "default_profile_id", "voices", "profiles"} or not isinstance(source["voices"], list) or not isinstance(source["profiles"], list):
        raise RoutingConfigurationError("invalid_tts_profiles_json")
    selected = source["default_profile_id"] if profile_id in {None, "default"} else profile_id
    profile = next((item for item in source["profiles"] if isinstance(item, dict) and item.get("id") == selected), None)
    if profile is None or profile.get("provider") != "elevenlabs":
        raise RoutingConfigurationError("unknown_tts_profile")
    strategy = profile.get("strategy")
    if strategy not in {"text_to_dialogue_v3", "stitched_text_to_speech"} or not settings.elevenlabs_api_key:
        raise RoutingConfigurationError("invalid_tts_profile")
    voices = {voice.get("id"): voice for voice in source["voices"] if isinstance(voice, dict) and isinstance(voice.get("id"), str)}
    requested = {"interviewer": interviewer_voice_id or profile.get("default_interviewer_voice_id"), "guest": guest_voice_id or profile.get("default_guest_voice_id")}
    bindings: dict[str, str] = {}
    profile_bindings = profile.get("voice_bindings", {})
    for role, voice_id in requested.items():
        voice = voices.get(voice_id)
        raw = profile_bindings.get(voice_id) if isinstance(profile_bindings, dict) else None
        if not voice or role not in voice.get("roles", []) or not isinstance(raw, dict) or not raw.get("provider_voice_id"):
            raise RoutingConfigurationError("invalid_tts_voice_binding")
        bindings[role] = raw["provider_voice_id"]
    model_id = profile.get("model_id")
    max_scene_characters, max_scene_turns = int(profile.get("max_scene_characters", 0)), int(profile.get("max_scene_turns", 0))
    max_fragment_characters = int(profile.get("max_fragment_characters", 0))
    if strategy == "text_to_dialogue_v3" and (model_id != "eleven_v3" or not 1 <= max_scene_characters <= 2000 or not 1 <= max_scene_turns <= 10 or max_fragment_characters != 0):
        raise RoutingConfigurationError("invalid_v3_tts_profile")
    if strategy == "stitched_text_to_speech" and (model_id == "eleven_v3" or max_fragment_characters < 1 or max_scene_characters != 0 or max_scene_turns != 0):
        raise RoutingConfigurationError("invalid_stitched_tts_profile")
    snapshot_payload = {"profile": profile, "bindings": bindings}
    return ResolvedTTSSnapshot(str(selected), _revision("tts", snapshot_payload), "elevenlabs", strategy, model_id, str(profile.get("output_format", "mp3")), max_scene_characters, max_scene_turns, max_fragment_characters, MappingProxyType(bindings), int(profile.get("max_attempts", 1)), int(profile.get("context_max_request_ids", 0)), int(profile.get("context_max_age_seconds", 0)), profile.get("continuity_after_expiry"), int(profile.get("max_concurrent_requests", 1)))


def validate_profile_pair(llm: ResolvedExecutionSnapshot, tts: ResolvedTTSSnapshot) -> None:
    """Compatibility hook for persistence/API lanes; budget validation remains server-side."""
    if not llm.revision or not tts.revision:
        raise RoutingConfigurationError("incompatible_generation_profiles")


settings = Settings()
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
MAX_TEXT_CHARS = settings.max_text_chars
MAX_UPLOAD_BYTES = settings.max_upload_bytes
MAX_CONCURRENT_GENERATIONS = settings.max_concurrent_generations
