"""Script store and management endpoints."""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from podcast_worker.core.models import (
    FollowUpRequest,
    ScriptRequest,
    ScriptStoreRequest,
    SummaryRequest,
)
from podcast_worker.core.script_generator import (
    generate_script,
    generate_follow_up_questions,
    generate_script_summary,
    generate_script_outline,
)
from podcast_worker.core.dependencies import resolve_llm_key
from podcast_worker.main import state

router = APIRouter(tags=["scripts"])
def _outline_from_script(script: dict) -> dict:
    if script.get("outline"):
        return script["outline"]

    return {
        "title": script.get("title", "Untitled"),
        "sections": [
            {
                "segment_type": segment.get("segment_type", "content"),
                "topic": segment.get("subtopic") or segment.get("title") or f"Section {index + 1}",
                "title": segment.get("title"),
                "approx_duration_seconds": segment.get("approx_duration_seconds"),
            }
            for index, segment in enumerate(script.get("segments", []))
        ],
    }




@router.post("/api/services/generate-script")
async def generate_script_endpoint(req: ScriptRequest):
    """Generate a podcast script only. Returns the script JSON synchronously."""
    try:
        llm_key = resolve_llm_key(req, "provider")
        script = generate_script(
            topic=req.topic,
            bpm=req.bpm,
            duration_minutes=req.duration_minutes,
            provider=req.provider,
            api_key=llm_key,
            model=req.model,
        )
        return {"status": "ok", "script": script}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/services/generate-outline")
async def generate_outline_endpoint(req: ScriptRequest):
    """Generate only the script outline synchronously."""
    try:
        llm_key = resolve_llm_key(req, "provider")
        outline = generate_script_outline(
            topic=req.topic,
            bpm=req.bpm,
            duration_minutes=req.duration_minutes,
            provider=req.provider,
            api_key=llm_key,
            model=req.model,
        )
        return {"status": "ok", "outline": outline}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/services/scripts", status_code=201)
async def create_script(req: ScriptStoreRequest):
    """Generate a podcast script, store it, and return a script_id."""
    try:
        llm_key = resolve_llm_key(req, "provider")
        script = generate_script(
            topic=req.topic,
            bpm=req.bpm,
            duration_minutes=req.duration_minutes,
            provider=req.provider,
            api_key=llm_key,
            model=req.model,
        )

        script_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        state.scripts[script_id] = {
            "script_id": script_id,
            "topic": req.topic,
            "bpm": req.bpm,
            "duration_minutes": req.duration_minutes,
            "script": script,
            "created_at": now,
            "follow_up_questions": None,
            "summary": None,
        }

        return {
            "status": "ok",
            "script_id": script_id,
            "script": script,
            "created_at": now,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/services/scripts")
async def list_scripts():
    """List all stored scripts with metadata (no full script content)."""
    return {
        "scripts": [
            {
                "script_id": s["script_id"],
                "topic": s["topic"],
                "bpm": s["bpm"],
                "duration_minutes": s["duration_minutes"],
                "title": s["script"].get("title", "Untitled"),
                "created_at": s["created_at"],
                "has_follow_up": s["follow_up_questions"] is not None,
                "has_summary": s["summary"] is not None,
            }
            for s in state.scripts.values()
        ],
        "count": len(state.scripts),
    }


@router.get("/api/services/scripts/{script_id}")
async def get_script(script_id: str):
    """Retrieve a stored script by its script_id. Returns full script content."""
    entry = state.scripts.get(script_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Script {script_id} not found.")
    return entry


@router.get("/api/services/scripts/{script_id}/outline")
async def get_script_outline(script_id: str):
    """Retrieve the outline for a stored script without full script text."""
    entry = state.scripts.get(script_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Script {script_id} not found.")

    return {
        "script_id": script_id,
        "outline": _outline_from_script(entry["script"]),
    }


@router.post("/api/services/scripts/{script_id}/follow-up")
async def generate_follow_up(script_id: str, req: FollowUpRequest):
    """Generate follow-up questions for a stored script."""
    entry = state.scripts.get(script_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Script {script_id} not found.")

    try:
        llm_key = resolve_llm_key(req, "provider")
        questions = generate_follow_up_questions(
            topic=req.topic,
            script=entry["script"],
            provider=req.provider,
            api_key=llm_key,
            model=req.model,
        )
        entry["follow_up_questions"] = questions

        return {
            "status": "ok",
            "script_id": script_id,
            "follow_up_questions": questions,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/services/scripts/{script_id}/summary")
async def generate_summary(script_id: str, req: SummaryRequest):
    """Generate a summary for a stored script."""
    entry = state.scripts.get(script_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Script {script_id} not found.")

    try:
        llm_key = resolve_llm_key(req, "provider")
        summary = generate_script_summary(
            script=entry["script"],
            provider=req.provider,
            api_key=llm_key,
            model=req.model,
        )
        entry["summary"] = summary

        return {
            "status": "ok",
            "script_id": script_id,
            "summary": summary,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))