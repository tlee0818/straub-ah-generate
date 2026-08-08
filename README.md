# StraubAH Generate Backend

FastAPI service for generating durable, two-speaker podcast projects. The server owns provider credentials, purpose-based LLM routing, validation, ElevenLabs speech generation, audio assembly, beat mixing, persistence, retries, and artifact delivery. Clients receive only opaque profile and voice IDs.

## Quick start

```bash
cp .env.example .env
docker compose up --build api
curl http://127.0.0.1:8100/api/v1/health
```

Minimum production secrets:

```dotenv
PODCAST_AUTH_TOKEN=<openssl rand -hex 32>
PODCAST_OPENROUTER_API_KEY=<key>
PODCAST_ELEVENLABS_API_KEY=<key>
SERVER_NAME=<public hostname>
```

Non-secret LLM routes and voice defaults are committed under `podcast_worker/config/`.

## API

Interactive Swagger UI is available at `/docs`; ReDoc is at `/redoc`. The committed OpenAPI document is [`docs/openapi.json`](docs/openapi.json). Product endpoints require:

```http
Authorization: Bearer <PODCAST_AUTH_TOKEN>
```

## Architecture documentation

- [Database ER diagram](docs/database-er-diagram.md)
- [Agent orchestration](docs/agent-orchestration.md)
- [Public API / Swagger](docs/public-api.md)
- [End-to-end generation pipeline](docs/generation-pipeline.md)

## Verification

```bash
python -m pytest -q
```

The Docker image includes FFmpeg and the Python dependencies required for generation, ElevenLabs, and mixing.
