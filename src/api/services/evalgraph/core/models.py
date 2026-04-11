# core/models.py
from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class Priority(str, Enum):
    CORRECTNESS    = "correctness"
    EFFICIENCY     = "efficiency"
    TIME_COMPLEXITY = "time_complexity"
    CODE_STYLE     = "code_style"


class Severity(str, Enum):
    CRITICAL = "critical"
    MAJOR    = "major"
    MINOR    = "minor"
    INFO     = "info"


# ─────────────────────────────────────────────────────────────────────────────
# SUB-MODELS
# ─────────────────────────────────────────────────────────────────────────────

class Citation(BaseModel):
    source:    str
    rule:      str
    excerpt:   str
    category:  str
    language:  str
    relevance: float


class Teaching(BaseModel):
    doc_passage:   str
    ai_summary:    str
    key_points:    list[str]
    reference_url: Optional[str] = None


class QuizQuestion(BaseModel):
    question_type:   str = "mcq"
    question:        str
    options:         Optional[list[str]] = None
    answer:          str
    explanation:     str
    source_citation: str = ""


class ShapScore(BaseModel):
    feature: str
    value:   float
    impact:  float


class Issue(BaseModel):
    issue_id:    str
    priority:    str        # Priority.value
    severity:    str        # Severity.value
    category:    str
    title:       str
    description: str
    line_hint:   Optional[int] = None
    fix:         str
    shap_scores: list[dict]
    citations:   list[dict]
    ieee_ref:    str
    teaching:    Optional[dict] = None


class CategoryScore(BaseModel):
    category:     str
    score:        float
    grade:        str
    issues_count: int
    weight:       float


class DiffAnalysis(BaseModel):
    issues_fixed:    int
    issues_new:      int
    score_delta:     float
    improved_areas:  list[str]
    remaining_areas: list[str]


class StaticMetrics(BaseModel):
    # Core identity
    language: str

    # Line-level counts
    loc: int
    sloc: int
    comments: int
    blank_lines: int
    comment_ratio: float

    # Cyclomatic complexity
    cyclomatic_complexity: float
    cc_rank: str
    cc_severity: Severity
    cc_interpretation: str
    functions_cc: list[dict]

    # Maintainability (MI)
    maintainability_index: float
    mi_label: str
    mi_severity: Severity

    # Halstead metrics
    halstead_volume: float
    halstead_difficulty: float
    halstead_effort: float
    halstead_bugs: float
    halstead_time_sec: float

    # Other metrics
    avg_function_length: float
    max_nesting_depth: int
    long_functions: list[dict]


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    code:           str
    language:       str       = "python"
    api_key:        str
    include_teach:  bool      = False
    include_quiz:   bool      = False
    top_k:          int       = 5
    previous_score: Optional[float] = None
    previous_code:  Optional[str]   = None
    # optional audio input (base64 data URI bodyless, i.e. data:audio/webm;base64,<data> stripped)
    audio_base64:   Optional[str] = None
    audio_mime:     Optional[str] = None
    # optional textual answer for assessment mode
    text_answer:    Optional[str] = None
    # include question description so backend can extract expected keywords
    question_description: Optional[str] = None


class FeedbackResponse(BaseModel):
    request_id:          str
    language:            str
    static_metrics:      dict
    overall_score:       float
    category_scores:     list[dict]
    issues:              list[dict]
    positives:           list[str]
    summary:             str
    next_steps:          list[str]
    quiz:                Optional[list[dict]] = None
    diff_analysis:       Optional[dict]       = None
    # echo transcript and submitted text answer if present
    audio_transcript:    Optional[str] = None
    keyword_evaluation:  Optional[list] = None
    text_answer:         Optional[str] = None
    processing_time_ms:  float


class RebuildRequest(BaseModel):
    api_key: str