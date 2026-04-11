# agents/graph.py  (final, clean)
from __future__ import annotations
from langgraph.graph import StateGraph, END
from .state import EvalState
from .nodes import (
    static_analysis_node, rag_retrieval_node, llm_review_node,
    issue_builder_node,   teaching_node,       quiz_node,
    diff_node,            assemble_response_node,
)


def _after_issues(state: EvalState) -> str:
    return "teaching" if state.get("include_teach") else "quiz"

def _after_quiz(state: EvalState) -> str:
    return "diff" if (
        state.get("previous_code") and state.get("previous_score") is not None
    ) else "assemble"


def build_graph() -> StateGraph:
    g = StateGraph(EvalState)

    for name, fn in [
        ("static_analysis", static_analysis_node),
        ("rag_retrieval",   rag_retrieval_node),
        ("llm_review",      llm_review_node),
        ("issue_builder",   issue_builder_node),
        ("teaching",        teaching_node),
        ("quiz",            quiz_node),
        ("diff",            diff_node),
        ("assemble",        assemble_response_node),
    ]:
        g.add_node(name, fn)

    g.set_entry_point("static_analysis")
    g.add_edge("static_analysis", "rag_retrieval")
    g.add_edge("rag_retrieval",   "llm_review")
    g.add_edge("llm_review",      "issue_builder")

    g.add_conditional_edges("issue_builder", _after_issues,
                            {"teaching": "teaching", "quiz": "quiz"})
    g.add_edge("teaching", "quiz")
    g.add_conditional_edges("quiz", _after_quiz,
                            {"diff": "diff", "assemble": "assemble"})
    g.add_edge("diff",     "assemble")
    g.add_edge("assemble", END)

    return g.compile()

def route_by_input(state):
    return "skip_metrics" if state.get("input_mode") in ("text", "audio") else "compute_metrics"

EVAL_GRAPH = build_graph()