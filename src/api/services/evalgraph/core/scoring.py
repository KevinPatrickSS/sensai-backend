from .models import Priority

IEEE_REFS: dict[str, str] = {
    "correctness":     "IEEE 730-2014 §6.3 — Software Quality Assurance: Correctness",
    "efficiency":      "IEEE 1061-1998 §4.2 — Software Quality Metrics: Efficiency",
    "time_complexity": "IEEE 1045-1992 — Software Productivity Metrics: Algorithm Complexity",
    "code_style":      "IEEE 1012-2016 §7 — Software Verification: Coding Standards",
    "best_practice":   "IEEE 730-2014 §6.5 — Software Quality Assurance: Maintainability",
    "lint":            "IEEE 1012-2016 §7.2 — Software V&V: Static Analysis",
    "style":           "IEEE 1016-2009 — Software Design Descriptions: Code Conventions",
    "security":        "IEEE 7009-2019 — Fail-Safe Design for Autonomous Systems",
}

SOURCE_URLS: dict[str, str] = {
    "PEP8":             "https://peps.python.org/pep-0008/",
    "CleanCode":        "https://www.oreilly.com/library/view/clean-code/9780136083238/",
    "ESLint":           "https://eslint.org/docs/latest/rules/",
    "Pylint":           "https://pylint.readthedocs.io/en/stable/",
    "Google Java Style":"https://google.github.io/styleguide/javaguide.html",
}

def grade_score(score: float) -> str:
    if score >= 90: return "A"
    if score >= 80: return "B"
    if score >= 70: return "C"
    if score >= 60: return "D"
    return "F"


PRIORITY_WEIGHTS = {
    Priority.CORRECTNESS:     0.40,
    Priority.EFFICIENCY:      0.25,
    Priority.TIME_COMPLEXITY: 0.20,
    Priority.CODE_STYLE:      0.15,
}

LLM_CAT_MAP = {
    "correctness":     Priority.CORRECTNESS,
    "efficiency":      Priority.EFFICIENCY,
    "time_complexity": Priority.TIME_COMPLEXITY,
    "code_style":      Priority.CODE_STYLE,
    "style":           Priority.CODE_STYLE,
    "lint":            Priority.TIME_COMPLEXITY,
    "best_practice":   Priority.EFFICIENCY,
    "best_practices":  Priority.EFFICIENCY,
    "security":        Priority.CORRECTNESS,
    "critical":        Priority.CORRECTNESS,
}
