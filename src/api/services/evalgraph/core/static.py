def compute_nesting_depth(code: str) -> int:
    """Count maximum nesting depth via indent heuristic."""
    max_depth = 0
    for line in code.splitlines():
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(stripped)
        depth  = indent // 4
        if depth > max_depth:
            max_depth = depth
    return max_depth


import re
from typing import List, Dict

from radon.raw import analyze
from radon.complexity import cc_visit, cc_rank
from radon.metrics import mi_visit, h_visit

from .models import StaticMetrics, Severity

# Threshold definitions for cyclomatic complexity ranks and MI labels
# Format for CC_THRESHOLDS: rank -> (low, high, Severity, interpretation)
CC_THRESHOLDS = {
    "A": (0, 5, Severity.INFO, "Low cyclomatic complexity"),
    "B": (6, 10, Severity.MINOR, "Moderate complexity"),
    "C": (11, 20, Severity.MAJOR, "High complexity"),
    "D": (21, 30, Severity.MAJOR, "Very high complexity"),
    "E": (31, 40, Severity.CRITICAL, "Extremely high complexity"),
    "F": (41, 999, Severity.CRITICAL, "Unmaintainable complexity"),
}

# MI_THRESHOLDS: (lo, hi, label, Severity)
MI_THRESHOLDS = [
    (0.0, 49.9, "Low", Severity.CRITICAL),
    (50.0, 64.9, "Moderate", Severity.MAJOR),
    (65.0, 84.9, "Good", Severity.MINOR),
    (85.0, 100.0, "Excellent", Severity.INFO),
]


def detect_long_functions(code: str, threshold: int = 30) -> list[dict]:
    """Return functions exceeding threshold lines."""
    results = cc_visit(code)
    long = []
    for fn in results:
        length = (fn.endline or fn.lineno) - fn.lineno
        if length > threshold:
            long.append({
                "name":      fn.name,
                "start_line": fn.lineno,
                "end_line":   fn.endline,
                "length":     length,
            })
    return long

def run_static_analysis(code: str, language: str) -> StaticMetrics:
    """
    Run full static analysis using radon.
    Falls back gracefully for non-Python code.
    """
    is_python = language.lower() == "python"

    # Defaults for non-Python
    loc = sloc = comments = blank = 0
    cc_avg = mi = h_volume = h_diff = h_effort = h_bugs = h_time = 0.0
    avg_fn_len = 0.0
    functions_cc: list[dict] = []
    long_fns: list[dict] = []
    max_nest = compute_nesting_depth(code)

    lines = code.splitlines()
    loc   = len(lines)

    if is_python:
        try:
            raw     = analyze(code)
            loc     = raw.loc
            sloc    = raw.sloc
            comments= raw.comments
            blank   = raw.blank
        except Exception:
            sloc = loc

        # Cyclomatic complexity
        try:
            cc_results = cc_visit(code)
            if cc_results:
                complexities = [r.complexity for r in cc_results]
                cc_avg = round(sum(complexities) / len(complexities), 2)
                for r in cc_results:
                    functions_cc.append({
                        "name":       r.name,
                        "complexity": r.complexity,
                        "rank":       cc_rank(r.complexity),
                        "line":       r.lineno,
                        "end_line":   r.endline,
                    })
                long_fns = detect_long_functions(code)
                if functions_cc:
                    avg_fn_len = round(
                        sum((f["end_line"] - f["line"]) for f in functions_cc) / len(functions_cc), 1
                    )
        except Exception:
            cc_avg = 1.0

        # Maintainability Index
        try:
            mi = round(mi_visit(code, multi=True), 2)
        except Exception:
            mi = 100.0

        # Halstead
        try:
            h    = h_visit(code)
            ht   = h.total
            h_volume = round(ht.volume,     2)
            h_diff   = round(ht.difficulty, 2)
            h_effort = round(ht.effort,     2)
            h_bugs   = round(ht.bugs,       3)
            h_time   = round(ht.time,       2)
        except Exception:
            pass
    else:
        # Basic metrics for non-Python
        sloc    = sum(1 for l in lines if l.strip() and not l.strip().startswith(("//","#","/*","*")))
        comments= sum(1 for l in lines if l.strip().startswith(("//","#","/*","*")))
        blank   = sum(1 for l in lines if not l.strip())

    # CC rank & thresholds
    overall_rank = cc_rank(int(cc_avg)) if cc_avg else "A"
    _, _, cc_sev, cc_interp = CC_THRESHOLDS.get(overall_rank, ("","",Severity.INFO,""))

    # MI thresholds
    mi_label = "Unknown"; mi_sev = Severity.INFO
    for lo, hi, label, sev in MI_THRESHOLDS:
        if lo <= mi <= hi:
            mi_label = label; mi_sev = sev; break

    comment_ratio = round(comments / max(loc, 1), 3)

    return StaticMetrics(
        language=language,
        loc=loc, sloc=sloc, comments=comments, blank_lines=blank,
        comment_ratio=comment_ratio,
        cyclomatic_complexity=cc_avg,
        cc_rank=overall_rank,
        cc_severity=cc_sev,
        cc_interpretation=cc_interp,
        functions_cc=functions_cc,
        maintainability_index=mi,
        mi_label=mi_label,
        mi_severity=mi_sev,
        halstead_volume=h_volume,
        halstead_difficulty=h_diff,
        halstead_effort=h_effort,
        halstead_bugs=h_bugs,
        halstead_time_sec=h_time,
        avg_function_length=avg_fn_len,
        max_nesting_depth=max_nest,
        long_functions=long_fns,
    )
