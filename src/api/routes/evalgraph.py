# src/api/routes/evalgraph.py
from __future__ import annotations
import os
import hashlib
import time
import base64
from fastapi import APIRouter, HTTPException

from api.services.evalgraph.agents.graph import EVAL_GRAPH
from api.services.evalgraph.agents.state import EvalState
from api.services.evalgraph.core.models import (
    FeedbackRequest,
    FeedbackResponse,
    RebuildRequest,
)
from api.services.evalgraph.core.retrieval import rebuild_index_logic
from api.services.evalgraph.core.llm import transcribe_audio
from api.services.evalgraph.core import memory as memory_store

router = APIRouter()


@router.on_event("startup")
async def startup_event():
    try:
        await memory_store.init_db()
    except Exception:
        pass


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest):
    if not req.api_key:
        req.api_key = os.getenv("OPENAI_API_KEY", "")
    t0         = time.time()
    request_id = hashlib.md5(f"{req.code}{t0}".encode("utf-8", "replace")).hexdigest()[:10]

    # Transcribe audio if provided
    audio_transcript = None
    if req.audio_base64:
        try:
            audio_bytes = base64.b64decode(req.audio_base64)
            audio_transcript = transcribe_audio(req.api_key, audio_bytes, req.audio_mime)
        except Exception as e:
            audio_transcript = f"[Transcription failed: {e}]"

    # Determine input_mode — drives graph routing
    if audio_transcript:
        input_mode = "audio"
    elif req.text_answer and not req.code.strip():
        input_mode = "text"
    else:
        input_mode = "code"

    initial_state: EvalState = {
        "code":                 req.code or "",
        "language":             req.language,
        "api_key":              req.api_key,
        "include_teach":        req.include_teach,
        "include_quiz":         req.include_quiz,
        "top_k":                req.top_k,
        "previous_score":       req.previous_score,
        "previous_code":        req.previous_code,
        "audio_transcript":     audio_transcript,
        "text_answer":          req.text_answer,
        "question_description": req.question_description,
        "input_mode":           input_mode,
        # intermediates start as None
        "metrics":              None,
        "retrieved":            None,
        "llm_result":           None,
        "raw_issues":           None,
        "teaching_map":         None,
        "quiz_data":            None,
        "diff_data":            None,
        "response":             None,
        "error":                None,
    }

    try:
        final_state = await EVAL_GRAPH.ainvoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if final_state.get("error"):
        raise HTTPException(status_code=500, detail=final_state["error"])

    result = final_state["response"]
    result["request_id"]         = request_id
    result["processing_time_ms"] = round((time.time() - t0) * 1000, 1)

    try:
        await memory_store.init_db()
        await memory_store.save_memory(
            request_id, result,
            tags=[result.get("language", ""), "feedback"]
        )
    except Exception:
        pass

    return FeedbackResponse(**result)


@router.post("/rebuild-index")
async def rebuild_index(req: RebuildRequest):
    return await rebuild_index_logic(req.api_key)


@router.get("/memories/{key}")
async def get_memory(key: str):
    try:
        m = await memory_store.get_memory(key)
        if m is None:
            raise HTTPException(status_code=404, detail="memory not found")
        return {"key": key, "value": m}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memories")
async def query_memories(tag: str | None = None):
    try:
        if tag:
            items = await memory_store.query_memories(tag)
            return {"count": len(items), "items": items}
        return {"count": 0, "items": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def evalgraph_health():
    from api.services.evalgraph.core.retrieval import RULES, EMBEDDINGS
    return {
        "status":      "ok",
        "rules_count": len(RULES),
        "index_dim":   int(EMBEDDINGS.shape[1]) if EMBEDDINGS.size else 0,
        "backend":     "sklearn NearestNeighbors",
    }