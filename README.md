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

## Deploy to a DigitalOcean Droplet

[`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) builds the API image, pushes the commit SHA and `latest` tags to GitHub Container Registry, and deploys the immutable SHA tag when `main` is updated. It also deploys Caddy from the committed [`Caddyfile`](Caddyfile), obtains and renews HTTPS certificates automatically, and verifies the public health endpoint. The workflow can also be run manually with **Actions → Deploy to DigitalOcean → Run workflow**.

The production API is available at `https://straubah.duckdns.org`. Caddy publishes ports 80 and 443 and proxies requests to the API over the private `straub-ah` Docker network. The workflow expects the production environment file at `/opt/straub-ah/.env`, keeps generated data in the `straub-ah-data` volume, and persists Caddy certificates in dedicated Docker volumes.

Configure these GitHub Actions environment secrets under **Settings → Environments → production → Environment secrets**:

| Secret | Value |
|---|---|
| `DROPLET_HOST` | Droplet IP address or DNS name |
| `DROPLET_USER` | SSH user with Docker access |
| `DROPLET_SSH_PRIVATE_KEY` | Private half of a dedicated deployment SSH key |
| `DROPLET_KNOWN_HOSTS` | Verified `known_hosts` line for the Droplet |
| `DROPLET_SSH_PORT` | Optional SSH port; defaults to `22` |

`GITHUB_TOKEN` is created automatically by GitHub Actions; do not add it manually. The workflow uses it to publish and pull this repository's GHCR image.

Generate a dedicated key, authorize it on the Droplet, and collect the host key:

```bash
ssh-keygen -t ed25519 -C "github-actions-straub-ah" -f ./straub-ah-deploy
ssh-copy-id -i ./straub-ah-deploy.pub <user>@<droplet-host>
ssh-keyscan -H <droplet-host>
```

Put the contents of `straub-ah-deploy` in `DROPLET_SSH_PRIVATE_KEY`. Put the verified `ssh-keyscan` output in `DROPLET_KNOWN_HOSTS`; compare its fingerprint with `ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub` from the DigitalOcean web console before trusting it. Delete the local private-key copy after storing it securely.

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
