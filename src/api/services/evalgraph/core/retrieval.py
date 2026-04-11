
import os
import json
import pickle
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss

from openai import OpenAI

from .models import StaticMetrics

def embed(client: OpenAI, texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = [item.embedding for item in resp.data]
    return np.array(vecs, dtype="float32")


RETRIEVAL_QUERIES = [
    "variable naming conventions best practices",
    "function length single responsibility principle",
    "error exception handling",
    "security hardcoded credentials secrets",
    "nested loops time complexity O(n squared)",
    "code readability documentation comments docstrings",
    "file resource management context manager",
    "type hints type safety",
    "global mutable state",
    "magic numbers named constants",
    "code duplication DRY principle",
    "logging production code",
]


def build_queries(code: str, language: str, metrics: StaticMetrics) -> list[str]:
    queries = list(RETRIEVAL_QUERIES)
    queries.append(f"{language} style guide coding standards")
    code_l = code.lower()

    if metrics.cyclomatic_complexity > 10:
        queries.append("cyclomatic complexity refactoring simplify")
    if metrics.max_nesting_depth > 3:
        queries.append("deeply nested code early return guard clause")
    if metrics.comment_ratio < 0.05:
        queries.append("missing comments documentation")
    if metrics.long_functions:
        queries.append("long function refactoring extract method")
    if re.search(r'for.+\n.+for', code, re.M):
        queries.append("nested loops O(n squared) optimization")
    if re.search(r'except\s*:', code):
        queries.append("bare except clause specific exceptions")
    if re.search(r'password|secret|api_key|token', code_l):
        queries.append("hardcoded credentials security vulnerability")
    if 'global ' in code_l:
        queries.append("global variables avoid mutable state")
    if re.search(r'open\s*\(', code_l) and 'with ' not in code_l:
        queries.append("file handling context manager with statement")

    return queries


def retrieve_rules(
    client: OpenAI,
    code: str,
    language: str,
    metrics: StaticMetrics,
    top_k: int = 6,
) -> list[tuple[float, dict]]:
    """
    Multi-query retrieval over the FAISS index.
    Returns deduplicated (score, rule) pairs sorted by relevance.
    """
    # sanitize inputs to avoid ascii codec issues
    code = _sanitize_text(code or "")
    language = _sanitize_text(language or "")
    queries = build_queries(code, language, metrics)
    try:
        query_vecs = embed(client, queries)
    except Exception:
        # Embedding failed (e.g., API error or encoding issue) — return empty retrievals
        return []

    # Normalise for cosine similarity
    faiss.normalize_L2(query_vecs)

    # Guard against missing or empty FAISS index (e.g., no embeddings loaded)
    if index is None or not hasattr(index, "ntotal") or index.ntotal == 0:
        # No index available — return empty retrievals to fail gracefully
        return []

    seen:   dict[int, float] = {}   # idx → best score
    k_per  = min(4, max(1, index.ntotal))

    for qvec in query_vecs:
        try:
            D, I = index.search(qvec.reshape(1, -1), k_per)
        except Exception:
            # Skip this query if the search fails for any reason
            continue
        for dist, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            score = float(dist)       # already cosine after normalisation
            if idx not in seen or score > seen[idx]:
                seen[idx] = score

    # Filter by language relevance
    lang_lower = language.lower()
    results: list[tuple[float, dict]] = []
    for idx, score in seen.items():
        rule = RULES[idx]
        rl   = rule.get("language", "general").lower()
        if rl not in ("general", lang_lower):
            score *= 0.6    # penalise wrong-language rules but keep them
        if score > 0.15:
            results.append((score, rule))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]


# core/retrieval.py
# ── paste embed(), build_queries(), retrieve_rules(),
#    EMBEDDINGS / NN_INDEX bootstrap block here ──
#
# PATH CONFIG — works locally AND on AWS:

def _resolve_embeddings_path() -> Path:
    """
    Priority:
      1. EMBEDDINGS_PATH env var  (set this in ECS task definition / Lambda env)
      2. /mnt/efs/evalgraph/       (EFS mount — recommended for ECS Fargate)
      3. /tmp/                     (Lambda ephemeral — copy from S3 on cold start)
      4. ./                        (local dev fallback)
    """
    if p := os.environ.get("EMBEDDINGS_PATH"):
        return Path(p)
    efs = Path("/mnt/efs/evalgraph")
    if efs.exists():
        return efs
    if Path("/tmp").exists() and os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        _maybe_download_from_s3()
        return Path("/tmp")
    return Path(__file__).parent.parent   # local dev: project root


def _maybe_download_from_s3():
    """Cold-start download for Lambda — only runs once per container."""
    import boto3
    bucket  = os.environ.get("S3_BUCKET", "evalgraph-artifacts")
    s3      = boto3.client("s3")
    for key, dest in [
        ("embeddings/rules_embeddings.pkl", "/tmp/rules_embeddings.pkl"),
        ("embeddings/rag_code_rules.json",  "/tmp/rag_code_rules.json"),
    ]:
        if not Path(dest).exists():
            s3.download_file(bucket, key, dest)


BASE_DIR        = _resolve_embeddings_path()
RULES_JSON      = BASE_DIR / "rag_code_rules.json"
EMBEDDINGS_FILE = BASE_DIR / "rules_embeddings.pkl"


# Load rules and embeddings / build FAISS index (graceful fallbacks)
if RULES_JSON.exists():
    with open(RULES_JSON, "r", encoding="utf-8") as f:
        RULES = json.load(f)
else:
    RULES = []

# Ensure textual fields in RULES are UTF-8 safe to avoid implicit ascii encoding errors
def _sanitize_text(s: str) -> str:
    if not isinstance(s, str):
        return str(s)
    try:
        return s.encode("utf-8", "replace").decode("utf-8")
    except Exception:
        return s

for r in RULES:
    if not isinstance(r, dict):
        continue
    for k in ("rule", "description", "source", "language", "category", "title"):
        if k in r and r[k] is not None:
            r[k] = _sanitize_text(r[k])


EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")

# Attempt to load a prebuilt embeddings matrix and create an inner-product index
index = None
try:
    if EMBEDDINGS_FILE.exists():
        with open(EMBEDDINGS_FILE, "rb") as f:
            emb = pickle.load(f)
        # emb may be a numpy array or dict with 'vectors'
        if isinstance(emb, dict):
            vecs = np.array(emb.get("vectors") or emb.get("embeddings") or emb.get("vecs"), dtype="float32")
        else:
            vecs = np.array(emb, dtype="float32")

        if vecs.ndim == 2 and vecs.shape[0] > 0:
            faiss.normalize_L2(vecs)
            dim = vecs.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(vecs)
        else:
            index = faiss.IndexFlatIP(1)  # empty placeholder
except Exception:
    index = faiss.IndexFlatIP(1)

# Ensure `index` is always set to a FAISS index object to avoid NoneType errors
if index is None:
    try:
        index = faiss.IndexFlatIP(1)
    except Exception:
        # As a last resort leave index as None; callers should handle it.
        index = None

# Expose embeddings matrix for health checks
try:
    EMBEDDINGS = vecs
except NameError:
    EMBEDDINGS = np.empty((0, 1), dtype="float32")


async def rebuild_index_logic(api_key: str) -> dict:
    """Rebuild embeddings + FAISS index from `RULES` and persist to EMBEDDINGS_FILE.

    Returns a status dict for the API endpoint.
    """
    client = OpenAI(api_key=api_key)
    texts = [r.get("description", "")[:2000] or r.get("rule", "") for r in RULES]
    if not texts:
        return {"status": "no_rules", "count": 0}

    # Batch embeddings (safe small batch)
    batch_size = 16
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs_b = [item.embedding for item in resp.data]
        all_vecs.extend(vecs_b)

    mat = np.array(all_vecs, dtype="float32")
    if mat.ndim != 2 or mat.shape[0] == 0:
        return {"status": "failed", "reason": "empty embeddings"}

    # Normalise and build index
    faiss.normalize_L2(mat)
    dim = mat.shape[1]
    new_index = faiss.IndexFlatIP(dim)
    new_index.add(mat)

    # Persist embeddings
    try:
        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(mat, f)
    except Exception as e:
        return {"status": "failed", "reason": f"persist_error: {e}"}

    # Update module-level state
    global index, EMBEDDINGS
    index = new_index
    EMBEDDINGS = mat

    return {"status": "ok", "count": mat.shape[0], "dim": int(mat.shape[1])}