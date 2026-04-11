import os
import re
import json
import textwrap
from typing import Any

from openai import OpenAI

from .models import StaticMetrics

CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")


def _llm(client: OpenAI, system: str, user: str, max_tokens: int = 2000) -> str:
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.15,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return resp.choices[0].message.content.strip()


def _json_llm(client: OpenAI, system: str, user: str, max_tokens: int = 2000) -> Any:
    raw = _llm(client, system, user, max_tokens)
    raw = re.sub(r"^```(?:json)?\n?", "", raw)
    raw = re.sub(r"\n?```$",          "", raw)
    return json.loads(raw)


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT: Code review (existing path)
# ─────────────────────────────────────────────────────────────────────────────
def build_review_prompt(
  code: str,
  language: str,
  metrics: StaticMetrics,
  retrieved: list[tuple[float, dict]],
  question_description: str | None = None,
  audio_transcript: str | None = None,
  text_answer: str | None = None,
  web_results: list[dict] | None = None,
) -> tuple[str, str]:

    doc_context = ""
    for i, (score, rule) in enumerate(retrieved, 1):
        excerpt = rule.get("description", "")[:300].replace("\n", " ")
        doc_context += (
            f"\n[DOC_{i}] Source={rule['source']} | Rule={rule['rule']} "
            f"| Category={rule['category']} | Lang={rule['language']}\n"
            f"  Excerpt: {excerpt}\n"
        )

    metrics_summary = (
        f"Cyclomatic Complexity: {metrics.cyclomatic_complexity} (rank {metrics.cc_rank})\n"
        f"Maintainability Index: {metrics.maintainability_index} ({metrics.mi_label})\n"
        f"Halstead Difficulty: {metrics.halstead_difficulty} | Volume: {metrics.halstead_volume}\n"
        f"Halstead Estimated Bugs: {metrics.halstead_bugs}\n"
        f"Max Nesting Depth: {metrics.max_nesting_depth}\n"
        f"Avg Function Length: {metrics.avg_function_length} lines\n"
        f"Comment Ratio: {metrics.comment_ratio}\n"
        f"Long Functions: {[f['name'] for f in metrics.long_functions]}\n"
    )

    web_context = ""
    if web_results:
        for i, r in enumerate(web_results, 1):
            web_context += f"\n[WEB_{i}] {r.get('title','')} | {r.get('url','')}\n  {r.get('snippet','')[:300]}\n"

    system = textwrap.dedent(f"""
    You are a strict tutor and senior code reviewer. Evaluate the student's submission with academic rigor.
    You have been given:
    1. Static analysis metrics already computed (DO NOT recompute these)
    2. Official documentation passages retrieved via semantic search (RAG)
    3. The problem description (use it to judge correctness)

    Priority order for issues (highest → lowest):
      1. CORRECTNESS    — bugs, logic errors, crashes
      2. EFFICIENCY     — algorithmic waste, N+1, redundant computation
      3. TIME_COMPLEXITY— O(n²)+ complexity, nested loops, sorting issues
      4. CODE_STYLE     — naming, formatting, documentation, lint

    IEEE standards to apply:
      - IEEE 730-2014  : Software Quality Assurance
      - IEEE 1061-1998 : Software Quality Metrics
      - IEEE 1012-2016 : Software V&V (static analysis)
      - IEEE 1045-1992 : Productivity/Complexity Metrics

    RETRIEVED OFFICIAL DOCUMENTATION:
    {doc_context}

    WEB SEARCH RESULTS (top):
    {web_context}

    STATIC ANALYSIS RESULTS (already computed — reference them):
    {metrics_summary}

    PROBLEM DESCRIPTION:
    {question_description or '<none>'}

    If the student also provided an audio submission, the transcript is:
    {audio_transcript or '<none>'}

    If the student provided a textual note (non-code), it is:
    {text_answer or '<none>'}

    You should behave as a tutor: be strict, reference relevant documentation, and provide concrete next-step learning materials.

    Respond ONLY with a valid JSON object — no markdown fences, no preamble.

    Schema:
    {{
      "summary": "<2-3 sentence overall assessment referencing the metrics>",
      "overall_score": <float 0-100>,
      "category_scores": {{
        "correctness":     <int 0-100>,
        "efficiency":      <int 0-100>,
        "time_complexity": <int 0-100>,
        "code_style":      <int 0-100>
      }},
      "issues": [
        {{
          "priority":    "<correctness|efficiency|time_complexity|code_style>",
          "severity":    "<critical|major|minor|info>",
          "category":    "<string>",
          "title":       "<short title>",
          "description": "<specific description — reference the exact metric or line>",
          "line_hint":   "<the exact bad code snippet, max 80 chars, or null>",
          "fix":         "<concrete actionable fix>",
          "doc_ref":     "<[DOC_N] or null>",
          "doc_source":  "<PEP8|CleanCode|ESLint|Pylint|Google Java Style|null>"
        }}
      ],
      "positives": ["<what is done well>"],
      "next_steps": ["<specific action 1>", "<specific action 2>"]
    }}

    Rules:
    - 3-8 issues, sorted by priority (correctness first)
    - Scores must be STRICT — reflect real code quality
    - Every issue with severity critical/major must have a doc_ref
    - Reference the computed metrics explicitly in descriptions
    """).strip()

    user = f"Review this {language} code:\n\n{code}"
    return system, user


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT: Text / Audio answer review  (NO static metrics, AI keyword evaluation)
# ─────────────────────────────────────────────────────────────────────────────
def build_text_review_prompt(
    submission: str,
    input_mode: str,                    # "text" or "audio"
    language: str,
    question_description: str | None = None,
    web_results: list[dict] | None = None,
) -> tuple[str, str]:
    """Build prompts for evaluating a written or spoken answer (no code)."""

    web_context = ""
    if web_results:
        for i, r in enumerate(web_results, 1):
            web_context += f"\n[WEB_{i}] {r.get('title','')} | {r.get('url','')}\n  {r.get('snippet','')[:300]}\n"

    input_label = "spoken answer (transcribed from audio)" if input_mode == "audio" else "written answer"

    system = textwrap.dedent(f"""
    You are a strict academic tutor evaluating a student's {input_label}.
    There is NO code submission — the student answered in {'speech' if input_mode == 'audio' else 'text'}.

    Your job:
    1. Read the question/problem description carefully.
    2. Read the student's answer carefully.
    3. Evaluate whether the student has FULLY answered all aspects of the question — not just keyword matching,
       but real comprehension: correctness of concepts, completeness, accuracy, depth, and clarity.
    4. Identify specific gaps, misunderstandings, or missing points as "issues".
    5. Score honestly: a vague or incomplete answer that happens to contain keywords should score LOW.

    PROBLEM DESCRIPTION / QUESTION:
    {question_description or '<none provided>'}

    WEB SEARCH RESULTS (use for factual grounding):
    {web_context or '<none>'}

    IEEE / Academic Standards to apply where relevant:
      - IEEE 730-2014 : Software Quality Assurance
      - IEEE 1061-1998: Software Quality Metrics

    Scoring dimensions:
      - correctness:     Are the facts and concepts stated correctly?
      - efficiency:      Does the student explain trade-offs or optimisation considerations?
      - time_complexity: If relevant, does the student discuss algorithmic complexity correctly?
      - code_style:      Clarity, structure, and completeness of the explanation.

    For "keyword_evaluation": analyse each important concept/keyword from the question description.
    For each concept, determine if the student TRULY addressed it (not just mentioned it).
    This is semantic evaluation — a student saying "it uses memory well" when asked about Big-O is NOT sufficient.

    Respond ONLY with a valid JSON object — no markdown fences, no preamble.

    Schema:
    {{
      "summary": "<2-3 sentence overall assessment>",
      "overall_score": <float 0-100>,
      "category_scores": {{
        "correctness":     <int 0-100>,
        "efficiency":      <int 0-100>,
        "time_complexity": <int 0-100>,
        "code_style":      <int 0-100>
      }},
      "keyword_evaluation": [
        {{
          "keyword":   "<concept or keyword from the question>",
          "addressed": <true|false>,
          "quality":   "<not_mentioned|mentioned_only|partially_explained|fully_explained>",
          "comment":   "<brief tutor comment on how well this was addressed>"
        }}
      ],
      "issues": [
        {{
          "priority":    "<correctness|efficiency|time_complexity|code_style>",
          "severity":    "<critical|major|minor|info>",
          "category":    "<string>",
          "title":       "<short title of the gap or error>",
          "description": "<specific description of what is wrong or missing>",
          "line_hint":   null,
          "fix":         "<what the student should do or say to fix this>",
          "doc_ref":     null,
          "doc_source":  null
        }}
      ],
      "positives": ["<what the student did well>"],
      "next_steps": ["<specific study/improvement action 1>", "<specific action 2>"]
    }}

    Rules:
    - 2-6 issues, ordered by severity
    - Be honest: a student who missed key concepts should receive a low score (below 50)
    - Do NOT reward keyword presence alone; reward demonstrated understanding
    - If the student's answer is off-topic or empty, overall_score should be 0-20
    """).strip()

    user = f"Student's {input_label}:\n\n{submission or '<empty submission>'}"
    return system, user


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT: Teaching
# ─────────────────────────────────────────────────────────────────────────────
def build_teaching_prompt(rule: dict, issue_title: str) -> tuple[str, str]:
    doc_text = rule.get("description", "")
    source   = rule.get("source", "")
    rule_name= rule.get("rule",   "")

    system = textwrap.dedent(f"""
    You are a programming educator. The student's submission has an issue: "{issue_title}".
    You will teach them from the following official documentation passage.

    SOURCE: {source}
    RULE: {rule_name}
    OFFICIAL TEXT:
    {doc_text}

    Respond ONLY with a valid JSON object:
    {{
      "doc_passage": "<most relevant verbatim sentences from the official text, max 300 chars>",
      "ai_summary": "<your clear 2-sentence explanation of why this matters>",
      "key_points": ["<point 1>", "<point 2>", "<point 3>"]
    }}
    """).strip()

    user = f"Teach the student about: {issue_title}"
    return system, user


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT: Quiz generation
# ─────────────────────────────────────────────────────────────────────────────
def build_quiz_prompt(
    issues: list[dict],
    retrieved: list[tuple[float, dict]],
    question_description: str | None = None,
) -> tuple[str, str]:
    context_parts = []
    for score, rule in retrieved[:4]:
        context_parts.append(
            f"Source: {rule['source']} | Rule: {rule['rule']}\n"
            f"Text: {rule.get('description','')[:250]}"
        )
    context = "\n\n".join(context_parts) if context_parts else "(no RAG docs — use the question description)"

    issue_titles = [i.get("title","") for i in issues[:5]]

    system = textwrap.dedent(f"""
    You are a programming quiz generator. Generate quiz questions that test understanding
    of the specific issues found in the evaluation.

    OFFICIAL DOCUMENTATION PASSAGES:
    {context}

    QUESTION/PROBLEM DESCRIPTION (if available):
    {question_description or '<none>'}

    EVALUATION ISSUES TO QUIZ ON:
    {json.dumps(issue_titles)}

    Respond ONLY with a valid JSON array of 3-5 quiz questions:
    [
      {{
        "question_type": "mcq",
        "question": "<clear question>",
        "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
        "answer": "A",
        "explanation": "<why this answer, citing the doc>",
        "source_citation": "<Source name, Rule name>"
      }},
      {{
        "question_type": "fill_blank",
        "question": "<sentence with ___ to fill>",
        "options": null,
        "answer": "<the word/phrase>",
        "explanation": "<explanation>",
        "source_citation": "<Source name>"
      }},
      {{
        "question_type": "true_false",
        "question": "<statement>",
        "options": ["True", "False"],
        "answer": "True or False",
        "explanation": "<explanation citing the doc>",
        "source_citation": "<Source name>"
      }}
    ]
    """).strip()

    user = "Generate the quiz questions."
    return system, user


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT: Re-evaluation / diff
# ─────────────────────────────────────────────────────────────────────────────
def build_reeval_prompt(
    original_code: str,
    revised_code: str,
    previous_score: float,
    language: str,
    retrieved: list[tuple[float, dict]],
) -> tuple[str, str]:
    doc_context = "\n".join(
        f"[DOC_{i}] {r['source']}: {r['rule']}" for i, (_, r) in enumerate(retrieved, 1)
    )
    system = textwrap.dedent(f"""
    You are re-evaluating revised code. Compare original vs revised and provide a diff-aware assessment.

    PREVIOUS SCORE: {previous_score}/100
    LANGUAGE: {language}
    RELEVANT DOCS: {doc_context}

    Respond ONLY with valid JSON:
    {{
      "issues_fixed":     <int>,
      "issues_new":       <int>,
      "score_delta":      <float — positive = improvement>,
      "improved_areas":   ["<area>"],
      "remaining_areas":  ["<area>"],
      "new_score":        <float 0-100>,
      "commentary":       "<2-sentence assessment>"
    }}
    """).strip()

    user = (
        f"ORIGINAL CODE:\n{original_code}\n\n"
        f"REVISED CODE:\n{revised_code}"
    )
    return system, user


# ─────────────────────────────────────────────────────────────────────────────
# Audio transcription
# ─────────────────────────────────────────────────────────────────────────────
def transcribe_audio(api_key: str, audio_bytes: bytes, mime: str | None = None) -> str:
    """Transcribe audio bytes using OpenAI Whisper model and return text."""
    import io

    # Supported Whisper extensions
    SUPPORTED = {"flac", "m4a", "mp3", "mp4", "mpeg", "mpga", "oga", "ogg", "wav", "webm"}

    # MIME → extension map (browser MediaRecorder sends mime like "audio/webm;codecs=opus")
    MIME_TO_EXT = {
        "audio/webm":  "webm",
        "audio/ogg":   "ogg",
        "audio/oga":   "oga",
        "audio/mp4":   "mp4",
        "audio/mpeg":  "mpeg",
        "audio/mp3":   "mp3",
        "audio/flac":  "flac",
        "audio/wav":   "wav",
        "audio/x-wav": "wav",
        "audio/m4a":   "m4a",
        "video/webm":  "webm",
        "video/mp4":   "mp4",
    }

    # Strip codec params: "audio/webm;codecs=opus" → "audio/webm"
    base_mime = (mime or "").split(";")[0].strip().lower()
    ext = MIME_TO_EXT.get(base_mime)

    if not ext:
        # Fall back: take part after "/" and strip anything after ";"
        raw_ext = base_mime.split("/")[-1].split(";")[0].strip()
        ext = raw_ext if raw_ext in SUPPORTED else "webm"

    client = OpenAI(api_key=api_key)
    try:
        fp = io.BytesIO(audio_bytes)
        fp.name = f"audio.{ext}"
        resp = client.audio.transcriptions.create(model="whisper-1", file=fp)
        text = resp.text if hasattr(resp, "text") else getattr(resp, "transcript", None)
        if not text:
            text = (resp.get("text") if isinstance(resp, dict) else None) or str(resp)
        return (text or "").strip()
    except Exception as e:
        raise RuntimeError(f"transcription failed: {e}")