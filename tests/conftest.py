"""pytest configuration for the services test suite."""
import os
os.environ.setdefault("PODCAST_ALLOW_INSECURE_DEV_AUTH", "true")
import sys

# Add the services/ root so we can import from podcast_worker.core
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
