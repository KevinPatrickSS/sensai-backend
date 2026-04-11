# agents/state.py
from __future__ import annotations
from typing import Any, Optional
from typing_extensions import TypedDict

class EvalState(TypedDict):
    # ── inputs ────────────────────────────────────────────────
    code:                  str
    language:              str
    api_key:               str
    include_teach:         bool
    include_quiz:          bool
    top_k:                 int
    previous_score:        Optional[float]
    previous_code:         Optional[str]
    
    # ── input routing (REQUIRED for proper feedback routing) ──
    input_mode:            str                  # "code" | "text" | "audio" — MUST be set
    audio_transcript:      Optional[str]        # transcribed from audio_base64
    text_answer:           Optional[str]        # text submission
    question_description:  Optional[str]        # for RAG context and keyword evaluation

    # ── intermediate ──────────────────────────────────────────
    metrics:               Optional[dict]       # StaticMetrics as dict
    retrieved:             Optional[list]       # list[tuple[float, dict]]
    llm_result:            Optional[dict]       # raw JSON from GPT-4o
    raw_issues:            Optional[list[dict]] # issues with citations+shap attached
    teaching_map:          Optional[dict[str, Any]] # issue_id → Teaching dict
    quiz_data:             Optional[list[dict]]
    diff_data:             Optional[dict]

    # ── output ────────────────────────────────────────────────
    response:              Optional[dict]       # final FeedbackResponse dict
    error:                 Optional[str]

    