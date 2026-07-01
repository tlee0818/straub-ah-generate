#!/usr/bin/env bash
# 🎙️  Straub AH — Podcast Worker Launcher
#
# Starts the Python Podcast Worker Service on port 8100.
# Run this before launching the iOS app to enable worker-backed generation.
#
# The worker service is self-contained in services/podcast_worker/
# and runs as: python -m services.podcast_worker.main
#
# Usage:
#   ./services/start-worker.sh            # Start the worker
#   ./services/start-worker.sh --reload   # Start with auto-reload (dev mode)
#   ./services/start-worker.sh --port 8200  # Custom port

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PORT=8100
RELOAD=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            PORT="$2"
            shift 2
            ;;
        --reload)
            RELOAD="--reload"
            shift
            ;;
        --help)
            echo "Usage: $0 [--port PORT] [--reload]"
            echo ""
            echo "Starts the Podcast Worker Service on the specified port (default: 8100)."
            echo ""
            echo "Options:"
            echo "  --port PORT    Port to bind (default: 8100)"
            echo "  --reload       Enable auto-reload on code changes (dev mode)"
            echo "  --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--port PORT] [--reload]"
            exit 1
            ;;
    esac
done

# Check Python availability
if ! command -v python3 &> /dev/null; then
    echo "❌ python3 is not installed or not in PATH."
    exit 1
fi

# Check for requirements
if [ ! -f "$SCRIPT_DIR/requirements-api.txt" ]; then
    echo "❌ requirements-api.txt not found in services/"
    exit 1
fi

# Ensure uvicorn and dependencies are installed
echo "📦 Checking Python dependencies..."
python3 -c "import uvicorn" 2>/dev/null || {
    echo "Installing API dependencies..."
    pip install -r "$SCRIPT_DIR/requirements-api.txt"
}

python3 -c "import numpy" 2>/dev/null || {
    echo "Installing core dependencies..."
    pip install -r "$SCRIPT_DIR/requirements.txt"
}

# Navigate to services/ (so the podcast_worker package is importable)
cd "$SCRIPT_DIR"

echo ""
echo "🎙️  ╔════════════════════════════════════════╗"
echo "🎙️  ║   Straub AH — Podcast Worker Service  ║"
echo "🎙️  ╚════════════════════════════════════════╝"
echo ""
echo "   Service:    http://0.0.0.0:${PORT}"
echo "   Health:     http://localhost:${PORT}/api/services/health"
echo "   API Docs:   http://localhost:${PORT}/docs"
echo "   Output Dir: output/"
echo "   Reload:     $([ -n "$RELOAD" ] && echo 'ON (dev mode)' || echo 'OFF')"
echo ""
echo "   Press Ctrl+C to stop"
echo ""

exec python3 -m uvicorn podcast_worker.main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --log-level info \
    $RELOAD
