"""
tutor_engine.py — LLM-powered tutor using Ollama (local inference).

Performance strategy:
  - Lessons and questions are cached in memory after first generation.
  - When a lesson is requested, a question is pre-fetched in a background
    thread so it's ready by the time the student finishes reading.
  - Correct answers get instant feedback (no LLM wait).
  - Wrong answers get full LLM feedback with step-by-step correction
    (~5–15s) — worth the wait for real tutoring value.
  - evaluate_answer_stream() yields live thinking/writing tokens for
    a real-time "AI console" in the frontend.
"""

from __future__ import annotations

import json as _json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

import httpx

logger = logging.getLogger(__name__)

OLLAMA_URL    = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3:4b"

# System context for lesson generation and question generation
_TUTOR_CONTEXT = (
    "You are a concise SPM Mathematics tutor for secondary school students. "
    "Give short, clear answers only. Use $...$ for inline math and $$...$$ for display math. "
    "Never add unnecessary words.\n\n"
)

# System context for evaluation — tells model NOT to re-solve the problem
_EVAL_CONTEXT = (
    "You are grading a student's maths answer. "
    "The expected answer is already given — do NOT re-solve the problem yourself. "
    "Your job: (1) decide if the student is correct, (2) give one sentence of targeted "
    "feedback about their specific mistake, (3) show the step-by-step working. "
    "Be encouraging and specific. Use $...$ for inline math.\n\n"
)

_OPTIONS = {"temperature": 0.3}

# ---------------------------------------------------------------------------
# In-memory caches
# ---------------------------------------------------------------------------
_lesson_cache:   dict[str, str]   = {}   # key → markdown lesson
_question_cache: dict[str, dict]  = {}   # concept_name → {question, hint, answer, _stale?}
_cache_lock = threading.Lock()

_prefetch_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="prefetch")


# ---------------------------------------------------------------------------
# Core LLM call (streaming, think=True keeps connection alive)
# ---------------------------------------------------------------------------

def _chat(user_content: str, model: str = DEFAULT_MODEL, system: str = _TUTOR_CONTEXT) -> str:
    """
    Call Ollama via streaming HTTP.  think=True routes reasoning into the
    'thinking' field; we collect only the 'content' tokens (the actual answer).
    Accepts an optional custom system message for different use cases.
    """
    body = {
        "model": model,
        "stream": True,
        "think": True,
        "options": _OPTIONS,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_content},
        ],
    }
    try:
        parts: list[str] = []
        with httpx.stream("POST", OLLAMA_URL, json=body, timeout=300.0) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                token = chunk.get("message", {}).get("content", "")
                if token:
                    parts.append(token)
                if chunk.get("done"):
                    break
        return "".join(parts).strip()
    except httpx.ConnectError:
        raise RuntimeError("Cannot reach Ollama. Is it running? (ollama serve)")
    except Exception as exc:
        logger.error("Ollama call failed: %s", exc)
        raise


def _chat_stream(
    user_content: str,
    model: str = DEFAULT_MODEL,
    system: str = _TUTOR_CONTEXT,
) -> Iterator[tuple[str, str]]:
    """
    Like _chat but yields (field, token) tuples so callers can stream
    thinking and content tokens separately.
      field == "thinking"  →  model's internal reasoning
      field == "content"   →  model's actual answer
    """
    body = {
        "model": model,
        "stream": True,
        "think": True,
        "options": _OPTIONS,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_content},
        ],
    }
    try:
        with httpx.stream("POST", OLLAMA_URL, json=body, timeout=300.0) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                msg = chunk.get("message", {})
                thinking_tok = msg.get("thinking", "")
                content_tok  = msg.get("content",  "")
                if thinking_tok:
                    yield ("thinking", thinking_tok)
                if content_tok:
                    yield ("content", content_tok)
                if chunk.get("done"):
                    break
    except httpx.ConnectError:
        raise RuntimeError("Cannot reach Ollama. Is it running? (ollama serve)")
    except Exception as exc:
        logger.error("Ollama stream failed: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Lesson generation (cached)
# ---------------------------------------------------------------------------

def generate_lesson(concept_name: str, prereq_names: list[str], model: str = DEFAULT_MODEL) -> str:
    """Return a cached markdown lesson, generating it on first call."""
    key = concept_name + "|" + ",".join(prereq_names)
    with _cache_lock:
        if key in _lesson_cache:
            return _lesson_cache[key]

    prereq_str = (
        f"The student already knows: {', '.join(prereq_names)}. "
        if prereq_names else "This is the student's first maths topic. "
    )
    prompt = (
        f"{prereq_str}"
        f"Teach **{concept_name}** for SPM Mathematics.\n\n"
        "Structure your lesson in markdown with these sections:\n"
        "## Definition\n"
        "One clear sentence.\n\n"
        "## Key Formula\n"
        "Show the formula and explain each variable. Use $$...$$ for display math.\n\n"
        "## Worked Example\n"
        "One concrete example with numbered steps. Show full working.\n\n"
        "## Common Mistake\n"
        "One specific misconception students make and how to avoid it.\n\n"
        "## Remember\n"
        "2–3 bullet tips.\n\n"
        "Keep under 350 words. Use $...$ for inline math, $$...$$ for display equations."
    )
    result = _chat(prompt, model)
    with _cache_lock:
        _lesson_cache[key] = result
    return result


# ---------------------------------------------------------------------------
# Question generation (cached, pre-fetchable)
# ---------------------------------------------------------------------------

def generate_question(
    concept_name: str,
    mastery_level: float = 0.5,
    model: str = DEFAULT_MODEL,
    force_refresh: bool = False,
) -> dict:
    """
    Return a cached question dict: {question, hint, answer}.
    If the cache entry is marked _stale=True (after a wrong answer),
    generate a fresh question.
    """
    # Cache key includes mastery bucket so crossing a threshold generates a harder question
    if mastery_level < 0.40:
        bucket = "basic"
    elif mastery_level < 0.70:
        bucket = "standard"
    else:
        bucket = "advanced"
    key = f"{concept_name}:{bucket}"

    with _cache_lock:
        cached = _question_cache.get(key)
        if cached and not force_refresh and not cached.get("_stale"):
            return cached

    mastery_pct = round(mastery_level * 100, 1)

    if mastery_level < 0.40:
        difficulty_instruction = (
            f"The student's current mastery of this topic is {mastery_pct}%. "
            "Generate a BASIC fundamental question testing a single definition or formula. "
            "One step, integer answer, no edge cases."
        )
    elif mastery_level < 0.70:
        difficulty_instruction = (
            f"The student's current mastery of this topic is {mastery_pct}%. "
            "Generate a STANDARD application question requiring 2–3 steps. "
            "Use realistic numbers; test procedural understanding."
        )
    else:
        difficulty_instruction = (
            f"The student's current mastery of this topic is {mastery_pct}%. "
            "Generate a HIGHLY COMPLEX edge-case question (Desirable Difficulty). "
            "Multi-step reasoning, non-obvious setup, or a common misconception as a trap. "
            "The student must demonstrate deep conceptual understanding to answer correctly."
        )

    prompt = (
        f"{difficulty_instruction}\n\n"
        f"Topic: '{concept_name}' (SPM Mathematics).\n"
        "Reply in this exact format — three lines only, nothing else:\n"
        "QUESTION: <the question, use $...$ for math>\n"
        "HINT: <a one-sentence hint guiding the method, not the answer>\n"
        "ANSWER: <the complete correct answer with units if needed>"
    )
    raw = _chat(prompt, model)
    result = _parse_question(raw)
    with _cache_lock:
        _question_cache[key] = result   # fresh entry, no _stale flag
    return result


def invalidate_question(concept_name: str) -> None:
    """
    Mark the cached question as stale so the NEXT GET /question call
    generates a fresh one.  The expected answer is preserved so that
    the current evaluation can still compare correctly.
    Marks all mastery buckets (basic/standard/advanced) for the concept.
    """
    with _cache_lock:
        for bucket in ("basic", "standard", "advanced"):
            key = f"{concept_name}:{bucket}"
            if key in _question_cache:
                _question_cache[key]["_stale"] = True
    logger.info("Marked all question buckets stale for '%s'", concept_name)


def prefetch_question(concept_name: str, mastery_level: float = 0.0) -> None:
    """
    Fire-and-forget: generate and cache a question in the background
    while the student is reading the lesson.
    """
    bucket = "basic" if mastery_level < 0.40 else ("standard" if mastery_level < 0.70 else "advanced")
    key = f"{concept_name}:{bucket}"
    with _cache_lock:
        cached = _question_cache.get(key)
        if cached and not cached.get("_stale"):
            return  # already cached and fresh
    _prefetch_pool.submit(_safe_prefetch, concept_name, mastery_level)


def _safe_prefetch(concept_name: str, mastery_level: float) -> None:
    try:
        generate_question(concept_name, mastery_level)
        logger.info("Prefetched question for '%s'", concept_name)
    except Exception as exc:
        logger.warning("Prefetch failed for '%s': %s", concept_name, exc)


# ---------------------------------------------------------------------------
# Evaluation (standard blocking version)
# ---------------------------------------------------------------------------

def _find_cached_answer(concept_name: str) -> str:
    """Search all mastery buckets for a cached expected answer."""
    with _cache_lock:
        for bucket in ("basic", "standard", "advanced"):
            entry = _question_cache.get(f"{concept_name}:{bucket}", {})
            if entry.get("answer"):
                return entry["answer"]
    return ""


def evaluate_answer(
    concept_name: str,
    question: str,
    student_answer: str,
    model: str = DEFAULT_MODEL,
) -> dict:
    """
    Evaluate a student's answer (blocking version).
      - Correct: instant feedback (no LLM).
      - Wrong: LLM call with expected answer provided (~5–15s).
      - No cache: full LLM evaluation.
    """
    expected = _find_cached_answer(concept_name)

    if expected:
        s_norm = _normalise(student_answer)
        e_norm = _normalise(expected)
        is_correct = (s_norm == e_norm) or _numbers_match(s_norm, e_norm)
        if is_correct:
            return {
                "score": 1.0,
                "is_correct": True,
                "feedback": "Correct! Great work.",
                "explanation": f"**Full working:** {expected}",
            }
        return _smart_evaluate(student_answer, expected, question, model)

    return _llm_evaluate(concept_name, question, student_answer, model)


def _smart_evaluate(
    student_answer: str,
    expected_answer: str,
    question: str,
    model: str = DEFAULT_MODEL,
) -> dict:
    """LLM evaluation when wrong — expected answer provided so model just explains."""
    prompt = (
        f"Question: {question}\n"
        f"Expected answer: {expected_answer}\n"
        f"Student wrote: {student_answer}\n\n"
        "Reply in this exact format — four lines only, nothing else:\n"
        "VERDICT: correct OR incorrect\n"
        "SCORE: 0-10\n"
        "FEEDBACK: <one sentence directly addressing what the student got wrong>\n"
        "SOLUTION: <full step-by-step working, 2–4 lines, use $...$ for math>"
    )
    raw = _chat(prompt, model, system=_EVAL_CONTEXT)
    result = _parse_evaluation(raw)
    result["is_correct"] = False
    if result["score"] >= 0.9:
        result["score"] = 0.5
    return result


def _llm_evaluate(
    concept_name: str,
    question: str,
    student_answer: str,
    model: str,
) -> dict:
    """Full LLM evaluation when no cached answer available."""
    prompt = (
        f"Topic: {concept_name}\n"
        f"Question: {question}\n"
        f"Student answer: {student_answer}\n\n"
        "Reply in this exact format — four lines only:\n"
        "VERDICT: correct OR incorrect\n"
        "SCORE: 0-10\n"
        "FEEDBACK: <one sentence directly addressing what the student wrote>\n"
        "SOLUTION: <step-by-step working, 2–4 lines, use $...$ for math>"
    )
    raw = _chat(prompt, model, system=_EVAL_CONTEXT)
    return _parse_evaluation(raw)


# ---------------------------------------------------------------------------
# Streaming evaluation — yields live thinking + writing tokens
# ---------------------------------------------------------------------------

def evaluate_answer_stream(
    concept_name: str,
    question: str,
    student_answer: str,
    model: str = DEFAULT_MODEL,
) -> Iterator[dict]:
    """
    Streaming version of evaluate_answer.
    Yields event dicts that the SSE endpoint can forward to the browser.

    Event shapes:
      {"phase": "checking",       "message": str}
      {"phase": "correct",        "result": dict}   ← instant, no LLM
      {"phase": "thinking_start", "message": str}
      {"phase": "thinking",       "token": str}     ← model's internal reasoning
      {"phase": "writing",        "token": str}     ← model writing the answer
      {"phase": "done",           "result": dict}   ← final evaluation result
      {"phase": "error",          "message": str}
    """
    expected = _find_cached_answer(concept_name)

    if expected:
        yield {"phase": "checking", "message": f"Comparing your answer to the expected solution…"}

        s_norm = _normalise(student_answer)
        e_norm = _normalise(expected)
        is_correct = (s_norm == e_norm) or _numbers_match(s_norm, e_norm)

        if is_correct:
            yield {
                "phase": "correct",
                "result": {
                    "score": 1.0,
                    "is_correct": True,
                    "feedback": "Correct! Great work.",
                    "explanation": f"**Full working:** {expected}",
                },
            }
            return

        # Wrong answer — stream LLM analysis
        yield {
            "phase": "thinking_start",
            "message": f"Answer doesn't match (expected: {expected}). Analysing your mistake…",
        }

        prompt = (
            f"Question: {question}\n"
            f"Expected answer: {expected}\n"
            f"Student wrote: {student_answer}\n\n"
            "Reply in this exact format — four lines only, nothing else:\n"
            "VERDICT: correct OR incorrect\n"
            "SCORE: 0-10\n"
            "FEEDBACK: <one sentence directly addressing what the student got wrong>\n"
            "SOLUTION: <full step-by-step working, 2–4 lines, use $...$ for math>"
        )

    else:
        # No cached answer — full evaluation
        yield {"phase": "thinking_start", "message": "No cached answer. Full evaluation starting…"}
        prompt = (
            f"Topic: {concept_name}\n"
            f"Question: {question}\n"
            f"Student answer: {student_answer}\n\n"
            "Reply in this exact format — four lines only:\n"
            "VERDICT: correct OR incorrect\n"
            "SCORE: 0-10\n"
            "FEEDBACK: <one sentence directly addressing what the student wrote>\n"
            "SOLUTION: <step-by-step working, 2–4 lines, use $...$ for math>"
        )

    # Stream the LLM call
    content_parts: list[str] = []
    try:
        for field, token in _chat_stream(prompt, model, system=_EVAL_CONTEXT):
            if field == "thinking":
                yield {"phase": "thinking", "token": token}
            elif field == "content":
                yield {"phase": "writing", "token": token}
                content_parts.append(token)
    except Exception as exc:
        yield {"phase": "error", "message": str(exc)}
        return

    raw = "".join(content_parts).strip()
    result = _parse_evaluation(raw)
    if not expected:
        pass  # let LLM decide is_correct
    else:
        result["is_correct"] = False
        if result["score"] >= 0.9:
            result["score"] = 0.5

    yield {"phase": "done", "result": result}


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_question(raw: str) -> dict:
    question = hint = answer = ""
    for line in raw.splitlines():
        line = line.strip()
        upper = line.upper()
        if upper.startswith("QUESTION:"):
            question = line.split(":", 1)[1].strip()
        elif upper.startswith("HINT:"):
            hint = line.split(":", 1)[1].strip()
        elif upper.startswith("ANSWER:"):
            answer = line.split(":", 1)[1].strip()
    if not question:
        question = raw.strip()
    return {"question": question, "hint": hint, "answer": answer}


def _parse_evaluation(raw: str) -> dict:
    """
    Parse the 4-line LLM evaluation response.
    SOLUTION: may span multiple lines — we capture everything after it.
    """
    verdict = score_str = feedback = ""
    solution_lines: list[str] = []
    in_solution = False

    for line in raw.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("VERDICT:"):
            verdict = stripped.split(":", 1)[1].strip().lower()
            in_solution = False
        elif upper.startswith("SCORE:"):
            score_str = stripped.split(":", 1)[1].strip()
            in_solution = False
        elif upper.startswith("FEEDBACK:"):
            feedback = stripped.split(":", 1)[1].strip()
            in_solution = False
        elif upper.startswith("SOLUTION:"):
            solution_lines = [stripped.split(":", 1)[1].strip()]
            in_solution = True
        elif in_solution and stripped:
            solution_lines.append(stripped)

    solution = "\n".join(solution_lines).strip()

    try:
        score = max(0.0, min(10.0, float(re.search(r"[\d.]+", score_str).group()))) / 10.0
    except Exception:
        score = 1.0 if "correct" in verdict and "incorrect" not in verdict else 0.15
    is_correct = ("incorrect" not in verdict) and ("correct" in verdict or score >= 0.7)
    return {
        "score": score,
        "is_correct": is_correct,
        "feedback": feedback or "Check your working.",
        "explanation": solution or "See the worked example in the lesson.",
    }


def _normalise(s: str) -> str:
    """Strip whitespace, lowercase, remove punctuation for comparison."""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[,;]", "", s)
    return s


def _numbers_match(a: str, b: str) -> bool:
    """Try to extract and compare the first number in each string."""
    def first_num(s: str):
        m = re.search(r"-?[\d]+(?:\.[\d]+)?", s)
        return float(m.group()) if m else None
    na, nb = first_num(a), first_num(b)
    return na is not None and nb is not None and abs(na - nb) < 1e-6
