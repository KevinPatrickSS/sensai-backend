from typing import List, Dict

from .models import StaticMetrics, ShapScore


def compute_shap_scores(code: str, metrics: StaticMetrics, issue_category: str) -> list[ShapScore]:
    """
    Lightweight SHAP-style attribution — which code features most contributed
    to this specific issue being flagged.
    """
    features: list[ShapScore] = []
    code_l = code.lower()

    feature_map = {
        "cyclomatic_complexity": (
            min(metrics.cyclomatic_complexity / 20, 1.0),
            "high" if metrics.cyclomatic_complexity > 10 else "low",
        ),
        "nesting_depth": (
            min(metrics.max_nesting_depth / 6, 1.0),
            "high" if metrics.max_nesting_depth > 3 else "low",
        ),
        "function_length": (
            min((metrics.avg_function_length or 0) / 50, 1.0),
            "high" if (metrics.avg_function_length or 0) > 30 else "low",
        ),
        "comment_ratio": (
            1.0 - metrics.comment_ratio,
            "low" if metrics.comment_ratio < 0.1 else "adequate",
        ),
        "halstead_difficulty": (
            min(metrics.halstead_difficulty / 30, 1.0),
            "high" if metrics.halstead_difficulty > 15 else "low",
        ),
        "maintainability_index": (
            max(0, 1.0 - metrics.maintainability_index / 100),
            "low" if metrics.maintainability_index < 65 else "adequate",
        ),
    }

    # Category-specific feature weights
    weights: dict[str, dict[str, float]] = {
        "efficiency":     {"cyclomatic_complexity":0.4, "nesting_depth":0.35, "halstead_difficulty":0.25},
        "time_complexity":{"cyclomatic_complexity":0.5, "nesting_depth":0.4,  "halstead_difficulty":0.1},
        "code_style":     {"comment_ratio":0.4,         "function_length":0.3, "nesting_depth":0.3},
        "best_practice":  {"comment_ratio":0.3,         "function_length":0.3, "maintainability_index":0.4},
        "correctness":    {"maintainability_index":0.4, "halstead_difficulty":0.35, "cyclomatic_complexity":0.25},
        "lint":           {"comment_ratio":0.5,         "nesting_depth":0.3, "function_length":0.2},
        "style":          {"comment_ratio":0.4,         "function_length":0.35, "nesting_depth":0.25},
    }

    cat_weights = weights.get(issue_category.lower(), weights["best_practice"])

    for feat, weight in cat_weights.items():
        base_val, direction = feature_map.get(feat, (0.0, "neutral"))
        contribution = round(base_val * weight, 3)
        # Map to ShapScore model fields: value (numeric) and impact (signed)
        impact = -1.0 if direction in ("high", "low") else 1.0
        features.append(ShapScore(
            feature=feat,
            value=contribution,
            impact=impact,
        ))

    features.sort(key=lambda x: x.value, reverse=True)
    return features