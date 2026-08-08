# Public API and Swagger

The canonical machine-readable contract is [`openapi.json`](openapi.json). It is OpenAPI 3.x JSON and can be imported directly into Swagger UI, Swagger Editor, Postman, Insomnia, or an SDK generator.

Runtime documentation:

- Swagger UI: `GET /docs`
- ReDoc: `GET /redoc`
- OpenAPI JSON: `GET /openapi.json`

Validate the committed document:

```bash
python -c 'import json; json.load(open("docs/openapi.json"))'
```

Regenerate it after changing routes or DTOs:

```bash
python -c 'import json; from podcast_worker.main import app; print(json.dumps(app.openapi(), indent=2, sort_keys=True))' > docs/openapi.json
```

## Authentication

Every product endpoint requires a bearer token. Health remains available for deployment probes.

```http
Authorization: Bearer <PODCAST_AUTH_TOKEN>
```

## Primary workflow

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | Deployment health and version. |
| `GET` | `/api/v1/config` | Safe opaque LLM/TTS profiles, role-eligible voices, and input ranges. |
| `POST` | `/api/v1/projects/outline-preview` | Generate and persist a reviewable, profile-bound outline. |
| `POST` | `/api/v1/projects` | Create a durable project and begin generation. |
| `GET` | `/api/v1/projects` | List canonical project manifests. |
| `GET` | `/api/v1/projects/{project_id}` | Fetch current project/segment/artifact state. |
| `GET` | `/api/v1/projects/{project_id}/outline` | Fetch the accepted section plan. |
| `DELETE` | `/api/v1/projects/{project_id}` | Apply the retention/deletion boundary. |
| `GET` | `/api/v1/artifacts/{artifact_id}` | Download an allowed public artifact. |
| `POST` | `/api/v1/artifacts/{artifact_id}/transfer-url` | Refresh a transfer URL when offload is enabled. |

## Trust boundary

Public payloads may contain only opaque profile/voice IDs. They never include provider API keys, raw ElevenLabs voice IDs, provider names, detailed model routes, pricing policies, or internal clip artifacts. Unknown or incompatible profile selections fail before provider work.

## Errors

Errors use a typed envelope with a stable code, message, and optional details. Important profile/preview cases include unavailable profiles, incompatible generation profiles, expired previews, missing legacy bindings, and preview/profile mismatch. Clients should refresh `/api/v1/config` and require a new preview rather than retrying against a silently changed profile.
