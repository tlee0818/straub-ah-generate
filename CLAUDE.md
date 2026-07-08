# StraubAH — Python Services

## Overview
The backend Python worker service for the StraubAH podcast generator. Provides a FastAPI HTTP API that iOS calls for heavy compute: LLM script generation, TTS synthesis, procedural beat generation, and audio mixing.

> **API Contract**: See [`API_SPEC.md`](../API_SPEC.md) in the root of the monorepo for the full request/response specification shared with the iOS client.

## Architecture

### Package Structure
```
services/
├── requirements.txt           # Core deps: openai, requests, click
├── requirements-api.txt       # API deps: fastapi, uvicorn, pydantic
├── start-worker.sh            # Launch script
├── podcast_worker/            # Python package (the service)
│   ├── main.py                # FastAPI app — all HTTP endpoints
│   └── core/
│       ├── config.py          # Configuration constants
│       ├── script_generator.py # LLM script generation (OpenAI/Ollama/OpenRouter)
│       ├── tts_engine.py      # TTS synthesis (edge-tts / OpenAI / OpenRouter)
│       ├── beat_generator.py  # Procedural beat synthesis (numpy)
│       └── audio_mixer.py     # Audio mixing with ducking (numpy)
└── tests/
    ├── conftest.py
    ├── test_beat_generator.py # 20+ tests for beat generation
    └── test_audio_mixer.py
```

### Canonical Product API

The hosted product API is `/api/v1/*`; see `../API_SPEC.md` for exact request and response schemas.

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/health` | GET | Health check |
| `/api/v1/config` | GET | Safe server-defined profiles, voices, and ranges |
| `/api/v1/projects/outline-preview` | POST | Generate a reviewable outline before full project creation |
| `/api/v1/projects` | POST | Create a durable project and start generation |
| `/api/v1/projects` | GET | List durable project manifests |
| `/api/v1/projects/{id}` | GET | Read the canonical project manifest |
| `/api/v1/projects/{id}/outline` | GET | Read the lightweight outline/section plan |
| `/api/v1/projects/{id}` | DELETE | Delete a project and retained artifacts |
| `/api/v1/artifacts/{id}` | GET | Download an artifact |
| `/api/v1/artifacts/{id}/transfer-url` | POST | Refresh a short-lived transfer URL |

Legacy `/api/services/*` endpoints remain dev-only and must not be used as product guidance.

## Tech Stack
- **Runtime**: Python 3.10+
- **Framework**: FastAPI + uvicorn
- **Audio**: numpy (beat synthesis, mixing), pydub (format conversion), wave (I/O)
- **LLM**: openai SDK (OpenAI, OpenRouter), requests (Ollama)
- **TTS**: edge-tts (free), openai SDK (OpenAI, OpenRouter)
- **FFmpeg**: required for MP3 conversion (`convert_to_mp3`)

## Generation Pipeline
1. **Outline preview** → `script_generator.py` → LLM returns JSON `{title, sections[]}` for user review
2. **Lead research agent** → creates episode-level brief, assumptions, and possible user follow-up questions
3. **Subtopic research agents** → gather section-specific points, examples, tensions, and cautions
4. **Dialogue writer agents** → interviewer and SME personas write approved outline sections with neighboring context and research packets
5. **Factfulness verifier** → checks/corrects each dialogue section against research context before TTS
6. **Speech** → `tts_engine.py` → TTS output (MP3), converted to WAV
7. **Beat** → `beat_generator.py` → numpy procedural beat at target BPM
8. **Mix** → `audio_mixer.py` → speech + beat with ducking, intro/outro, normalize

## Key Design Decisions
- **Durable v1 state** lives in SQLite-backed `PodcastProject`, segment, artifact, provenance, and error tables.
- **Outline-first generation** lets iOS show and refine the episode plan before full script/audio work starts.
- **Multi-agent script generation** separates research coordination, subtopic research, interviewer/SME dialogue writing, and factfulness verification so tone stays engaging without losing factual structure.
- **Legacy `/api/services/*` routes** are retained only for development experiments.

## Beat Engine (Python)
- Built with numpy (vs. Accelerate/vDSP in the Swift version)
- Same algorithm: kick, snare, hi-hat, sub bass, noise swell, sidechain compression
- Save to WAV via `wave` module (mono 16-bit)

## Audio Mixer
- Ducking: speech-triggered gain reduction on beat (-18dB during speech, -6dB baseline)
- Speech region detection: frame-based energy thresholding (20ms frames)
- Intro: beat-only with fade-in (2s)
- Outro: beat-only with fade-out (3s)
- Safety fades: 50ms at start and end
- MP3 conversion via ffmpeg (libmp3lame, 192k)

## Running
```bash
./services/start-worker.sh           # Port 8100
./services/start-worker.sh --reload  # Dev mode with auto-reload
```

## Configuration (`core/config.py`)
- LLM: OpenAI, Ollama, OpenRouter (model, API key, base URL)
- TTS: edge (free), OpenAI, OpenRouter (voice, model)
- BPM range: 60-220
- Sample rate: 44100

## Testing
```bash
cd services && python -m pytest tests/ -v
```
Tests cover beat generation (BPM conversion, energy profiles, amplitude, WAV output, edge cases).
