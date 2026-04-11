# agents/nodes.py
from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

from openai import OpenAI

from .state import EvalState

# ── import all logic from main.py (unchanged) ──────────────────────────────
from api.services.evalgraph.core.static    import run_static_analysis
from api.services.evalgraph.core.retrieval import retrieve_rules
from api.services.evalgraph.core.llm       import (
    build_review_prompt, build_teaching_prompt,
    build_quiz_prompt,   build_reeval_prompt,
    _json_llm,
)
from api.services.evalgraph.core.shap      import compute_shap_scores
from api.services.evalgraph.core.scoring   import (
    grade_score, PRIORITY_WEIGHTS, LLM_CAT_MAP,
    IEEE_REFS, SOURCE_URLS,
)
from api.services.evalgraph.core.models    import (
    Priority, Severity, Citation, Teaching,
    QuizQuestion, Issue, CategoryScore, DiffAnalysis,
    StaticMetrics,
)
from api.services.evalgraph.core.llm import (
    build_review_prompt,
    build_text_review_prompt,
    _json_llm,
)



def _stub_metrics(language: str) -> dict:
    """Return zero-filled metrics for non-code submissions."""
    return {
        "language":              language,
        "loc":                   0,
        "sloc":                  0,
        "comments":              0,
        "blank_lines":           0,
        "comment_ratio":         0.0,
        "cyclomatic_complexity": 0.0,
        "cc_rank":               "N/A",
        "cc_severity":           "info",
        "cc_interpretation":     "Not applicable for text/audio submission",
        "functions_cc":          [],
        "maintainability_index": 0.0,
        "mi_label":              "N/A",
        "mi_severity":           "info",
        "halstead_volume":       0.0,
        "halstead_difficulty":   0.0,
        "halstead_effort":       0.0,
        "halstead_bugs":         0.0,
        "halstead_time_sec":     0.0,
        "avg_function_length":   0.0,
        "max_nesting_depth":     0,
        "long_functions":        [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1 — static analysis  (skipped for text/audio submissions)
# ─────────────────────────────────────────────────────────────────────────────
def static_analysis_node(state: EvalState) -> dict:
    # Text/audio: return stub, no code to analyse
    if state.get("input_mode") in ("text", "audio"):
        return {"metrics": _stub_metrics(state["language"])}
    try:
        metrics = run_static_analysis(state["code"], state["language"])
        return {"metrics": metrics.__dict__}
    except Exception as e:
        return {"error": f"Static analysis failed: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2 — RAG retrieval  (skipped for text/audio submissions)
# ─────────────────────────────────────────────────────────────────────────────
def rag_retrieval_node(state: EvalState) -> dict:
    if state.get("error"):
        return {}
    # Text/audio: no code to embed; return empty retrieved list
    if state.get("input_mode") in ("text", "audio"):
        return {"retrieved": []}
    try:
        client   = OpenAI(api_key=state["api_key"])
        metrics  = _dict_to_metrics(state["metrics"])
        retrieved = retrieve_rules(
            client, state["code"], state["language"], metrics, top_k=state["top_k"]
        )
        return {"retrieved": [[float(s), r] for s, r in retrieved]}
    except Exception as e:
        return {"error": f"RAG retrieval failed: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3 — LLM review  (uses text/audio-specific prompt when applicable)
# ─────────────────────────────────────────────────────────────────────────────
def llm_review_node(state: EvalState) -> dict:
    if state.get("error"):
        return {}
    try:
        client    = OpenAI(api_key=state["api_key"])
        metrics   = _dict_to_metrics(state["metrics"])
        retrieved = [(s, r) for s, r in (state["retrieved"] or [])]

        web_results = []
        qdesc = state.get("question_description")
        if qdesc:
            try:
                web_results = web_search(qdesc, top_k=5)
            except Exception:
                web_results = []

        input_mode = state.get("input_mode", "code")
        if not state.get("input_mode"):
            import logging
            logging.warning(f"[DEBUG] input_mode was None/missing in llm_review_node, defaulting to 'code'. Full state keys: {list(state.keys())}")

        if input_mode in ("text", "audio"):
            # Use dedicated text/audio prompt that does AI-based keyword evaluation
            submission = state.get("audio_transcript") or state.get("text_answer") or ""
            sys_p, usr_p = build_text_review_prompt(
                submission=submission,
                input_mode=input_mode,
                language=state["language"],
                question_description=qdesc,
                web_results=web_results,
            )
        else:
            sys_p, usr_p = build_review_prompt(
                state["code"], state["language"], metrics, retrieved,
                question_description=qdesc,
                audio_transcript=state.get("audio_transcript"),
                text_answer=state.get("text_answer"),
                web_results=web_results,
            )

        result = _json_llm(client, sys_p, usr_p, max_tokens=3000)
        return {"llm_result": result}
    except Exception as e:
        return {"error": f"LLM review failed: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4 — issue builder
# ─────────────────────────────────────────────────────────────────────────────
def issue_builder_node(state: EvalState) -> dict:
    if state.get("error"):
        return {}

    metrics    = _dict_to_metrics(state["metrics"])
    retrieved  = [(s, r) for s, r in (state["retrieved"] or [])]
    rule_map   = {i + 1: r for i, (_, r) in enumerate(retrieved)}
    score_map  = {i + 1: s for i, (s, _) in enumerate(retrieved)}
    llm_result = state["llm_result"]
    input_mode = state.get("input_mode", "code")
    if not state.get("input_mode"):
        import logging
        logging.warning(f"[DEBUG] input_mode was None/missing in issue_builder_node, defaulting to 'code'. Full state keys: {list(state.keys())}")

    raw_issues: list[dict] = []
    for raw in llm_result.get("issues", []):
        cat_str  = raw.get("category", "best_practice")
        priority = LLM_CAT_MAP.get(raw.get("priority", "code_style").lower(), Priority.CODE_STYLE)
        severity = Severity(raw.get("severity", "minor"))
        ieee_ref = IEEE_REFS.get(raw.get("priority", "code_style").lower(), IEEE_REFS["code_style"])

        # SHAP only meaningful for code; return empty list for text/audio
        if input_mode == "code":
            shap = [s.__dict__ for s in compute_shap_scores(state["code"], metrics, cat_str)]
        else:
            shap = []

        doc_ref_str = raw.get("doc_ref") or ""
        doc_nums    = [int(x) for x in re.findall(r"DOC_(\d+)", doc_ref_str)] or [1]

        citations: list[dict] = []
        for dn in doc_nums[:2]:
            rule = rule_map.get(dn)
            if rule:
                citations.append({
                    "source":    rule.get("source", ""),
                    "rule":      rule.get("rule", "")[:80],
                    "excerpt":   rule.get("description", "")[:200],
                    "category":  rule.get("category", ""),
                    "language":  rule.get("language", ""),
                    "relevance": round(score_map.get(dn, 0.0), 3),
                })

        title_bytes = (raw.get("title") or "").encode("utf-8", "replace")
        issue_id = hashlib.md5(title_bytes).hexdigest()[:8]
        raw_issues.append({
            "issue_id":    issue_id,
            "priority":    priority.value,
            "severity":    severity.value,
            "category":    cat_str,
            "title":       raw.get("title", ""),
            "description": raw.get("description", ""),
            "line_hint":   raw.get("line_hint"),
            "fix":         raw.get("fix", ""),
            "shap_scores": shap,
            "citations":   citations,
            "ieee_ref":    ieee_ref,
            "_doc_nums":   doc_nums,
        })

    return {"raw_issues": raw_issues}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 5 — teaching
# ─────────────────────────────────────────────────────────────────────────────
def teaching_node(state: EvalState) -> dict:
    if state.get("error") or not state.get("include_teach"):
        return {"teaching_map": {}}

    client    = OpenAI(api_key=state["api_key"])
    retrieved = [(s, r) for s, r in (state["retrieved"] or [])]
    rule_map  = {i + 1: r for i, (_, r) in enumerate(retrieved)}

    teaching_map: dict[str, dict] = {}
    for issue in state.get("raw_issues", []):
        severity = Severity(issue["severity"])
        if severity not in (Severity.CRITICAL, Severity.MAJOR):
            continue
        doc_nums  = issue.get("_doc_nums", [1])
        citations = issue.get("citations", [])
        if not citations or not doc_nums:
            continue
        try:
            ref_rule     = rule_map[doc_nums[0]]
            t_sys, t_usr = build_teaching_prompt(ref_rule, issue["title"])
            t_data       = _json_llm(client, t_sys, t_usr, max_tokens=600)
            teaching_map[issue["issue_id"]] = {
                "doc_passage":   t_data.get("doc_passage", ""),
                "ai_summary":    t_data.get("ai_summary",  ""),
                "key_points":    t_data.get("key_points",  []),
                "reference_url": SOURCE_URLS.get(citations[0]["source"]),
            }
        except Exception:
            pass

    return {"teaching_map": teaching_map}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 6 — quiz generation
# ─────────────────────────────────────────────────────────────────────────────
def quiz_node(state: EvalState) -> dict:
    if state.get("error") or not state.get("include_quiz"):
        return {"quiz_data": None}

    client    = OpenAI(api_key=state["api_key"])
    retrieved = [(s, r) for s, r in (state["retrieved"] or [])]

    try:
        q_sys, q_usr = build_quiz_prompt(
            [{("title"): i["title"]} for i in state.get("raw_issues", [])],
            retrieved,
            question_description=state.get("question_description"),
        )
        q_data = _json_llm(client, q_sys, q_usr, max_tokens=1500)
        return {"quiz_data": q_data if isinstance(q_data, list) else None}
    except Exception:
        return {"quiz_data": None}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 7 — diff / re-evaluation
# ─────────────────────────────────────────────────────────────────────────────
def diff_node(state: EvalState) -> dict:
    if state.get("error"):
        return {"diff_data": None}
    if not state.get("previous_code") or state.get("previous_score") is None:
        return {"diff_data": None}

    client    = OpenAI(api_key=state["api_key"])
    retrieved = [(s, r) for s, r in (state["retrieved"] or [])]

    try:
        r_sys, r_usr = build_reeval_prompt(
            state["previous_code"], state["code"],
            state["previous_score"],  state["language"], retrieved,
        )
        r_data = _json_llm(client, r_sys, r_usr, max_tokens=600)
        return {"diff_data": r_data}
    except Exception:
        return {"diff_data": None}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 8 — assemble final response
# ─────────────────────────────────────────────────────────────────────────────
def assemble_response_node(state: EvalState) -> dict:
    if state.get("error"):
        return {"response": {"error": state["error"]}}

    llm_result   = state["llm_result"]
    raw_issues   = state.get("raw_issues", [])
    teaching_map = state.get("teaching_map", {})
    quiz_data    = state.get("quiz_data")
    diff_data    = state.get("diff_data")

    issues_out = []
    for iss in raw_issues:
        iss_out = {k: v for k, v in iss.items() if k != "_doc_nums"}
        iss_out["teaching"] = teaching_map.get(iss["issue_id"])
        issues_out.append(iss_out)

    pri_order = {"correctness": 0, "efficiency": 1, "time_complexity": 2, "code_style": 3}
    sev_order = {"critical": 0, "major": 1, "minor": 2, "info": 3}
    issues_out.sort(key=lambda i: (
        pri_order.get(i["priority"], 9),
        sev_order.get(i["severity"], 9),
    ))

    raw_cat = llm_result.get("category_scores", {})
    category_scores = []
    for priority, weight in PRIORITY_WEIGHTS.items():
        s     = float(raw_cat.get(priority.value, 70))
        count = sum(1 for i in issues_out if i["priority"] == priority.value)
        category_scores.append({
            "category":     priority.value,
            "score":        s,
            "grade":        grade_score(s),
            "issues_count": count,
            "weight":       weight,
        })

    overall = float(llm_result.get("overall_score", 70.0))

    diff_out = None
    if diff_data:
        overall = float(diff_data.get("new_score", overall))
        diff_out = {
            "issues_fixed":    diff_data.get("issues_fixed",    0),
            "issues_new":      diff_data.get("issues_new",      0),
            "score_delta":     diff_data.get("score_delta",     0.0),
            "improved_areas":  diff_data.get("improved_areas",  []),
            "remaining_areas": diff_data.get("remaining_areas", []),
        }

    quiz_out = None
    if quiz_data:
        quiz_out = [
            {
                "question_type":   q.get("question_type", "mcq"),
                "question":        q.get("question", ""),
                "options":         q.get("options"),
                "answer":          q.get("answer", ""),
                "explanation":     q.get("explanation", ""),
                "source_citation": q.get("source_citation", ""),
            }
            for q in quiz_data
        ]

    # Pull AI keyword evaluation from llm_result if present
    keyword_evaluation = llm_result.get("keyword_evaluation")
    
    # SAFETY: For text/audio submissions, keyword_evaluation MUST be present
    input_mode = state.get("input_mode")
    if input_mode in ("text", "audio") and (not keyword_evaluation or len(keyword_evaluation) == 0):
        import logging
        logging.warning(f"[SAFETY] Text/audio submission (mode={input_mode}) but keyword_evaluation is empty/missing from LLM result. This may indicate a backend routing issue.")

    return {
        "response": {
            "language":           state["language"],
            "static_metrics":     state["metrics"],
            "overall_score":      overall,
            "category_scores":    category_scores,
            "issues":             issues_out,
            "positives":          llm_result.get("positives",  []),
            "summary":            llm_result.get("summary",    ""),
            "next_steps":         llm_result.get("next_steps", []),
            "quiz":               quiz_out,
            "diff_analysis":      diff_out,
            "audio_transcript":   state.get("audio_transcript"),
            "text_answer":        state.get("text_answer"),
            "keyword_evaluation": keyword_evaluation,
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _dict_to_metrics(d: dict) -> Any:
    """Re-hydrate a dict back into a StaticMetrics-compatible namespace."""
    from types import SimpleNamespace
    return SimpleNamespace(**d)