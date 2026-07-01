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

### API Endpoints (port 8100)
| Endpoint | Method | Type | Description |
|---|---|---|---|
| `/api/services/health` | GET | Sync | Health check |
| `/api/services/config` | GET | Sync | Available providers, voices, models, BPM range |
| `/api/services/generate` | POST | Async | Full pipeline — returns `job_id` for polling |
| `/api/services/jobs/{id}` | GET | Sync | Poll job status/result |
| `/api/services/jobs/{id}/result` | GET | Sync | Download audio file |
| `/api/services/jobs/{id}/script` | GET | Sync | Download script JSON |
| `/api/services/generate-script` | POST | Sync | Script generation only |
| `/api/services/generate-audio` | POST | Sync | TTS + beat + mix (full audio) |
| `/api/services/generate-beat` | POST | Sync | Beat-only generation (returns WAV) |
| `/api/services/generate-speech` | POST | Sync | TTS-only generation (returns WAV) |
| `/api/services/overlay-audio` | POST | Sync | Multipart upload: speech WAV + beat WAV → mixed podcast |

## Tech Stack
- **Runtime**: Python 3.10+
- **Framework**: FastAPI + uvicorn
- **Audio**: numpy (beat synthesis, mixing), pydub (format conversion), wave (I/O)
- **LLM**: openai SDK (OpenAI, OpenRouter), requests (Ollama)
- **TTS**: edge-tts (free), openai SDK (OpenAI, OpenRouter)
- **FFmpeg**: required for MP3 conversion (`convert_to_mp3`)

## Generation Pipeline
1. **Script** → `script_generator.py` → LLM returns JSON `{title, segments[]}`
2. **Speech** → `tts_engine.py` → TTS output (MP3), converted to WAV
3. **Beat** → `beat_generator.py` → numpy procedural beat at target BPM
4. **Mix** → `audio_mixer.py` → speech + beat with ducking, intro/outro, normalize

## Key Design Decisions
- **Jobs stored in-memory** (simple dict). For production, swap to Redis/DB.
- **Separated pipeline** (`generate-speech` + `generate-beat` + `overlay-audio`) allows the iOS app to run steps independently and handle overlay locally if needed.
- **Async full pipeline** via `generate` endpoint runs in a daemon thread.
- **All sync endpoints** (`generate-beat`, `generate-speech`, `overlay-audio`) are synchronous — fast enough for real-time use.

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
