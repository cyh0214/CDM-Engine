"""
api.py — FastAPI layer for the Cognitive Decay Matrix (CDM).

Architecture contract:
  - Core math/DAG logic lives exclusively in decay_math.py and dag_engine.py.
  - This file is ONLY the I/O layer: HTTP ↔ DB ↔ Engine.
  - The engine is hydrated from SQLite on every request (stateless HTTP).
  - After a write, the updated state is persisted back to SQLite.
"""

import json as _api_json
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import os

# --- Persistence layer ---
from database import init_db, get_db, DBStudentState, SessionLocal

# --- Core engine (untouched) ---
from models import ConceptNode, StudentState
from dag_engine import KnowledgeGraph, StudentEngine, DECAY_THRESHOLD

# --- New adaptive tutor modules ---
from tutor_engine import (
    generate_lesson, generate_question, evaluate_answer_stream,
    prefetch_question, invalidate_question,
)
from knowledge_journey import get_journey, get_next_node


# ---------------------------------------------------------------------------
# Static Knowledge DAG — SPM Mathematics (9 nodes)
# Extend this list to add more subjects/nodes without touching the engine.
# ---------------------------------------------------------------------------
_GRAPH = KnowledgeGraph()
for _node in [
    ConceptNode("spm_01", "Algebra",         prerequisites=[]),
    ConceptNode("spm_02", "Quadratics",      prerequisites=["spm_01"]),
    ConceptNode("spm_03", "Calculus",        prerequisites=["spm_02"]),
    ConceptNode("spm_04", "Trigonometry",    prerequisites=["spm_01"]),
    ConceptNode("spm_05", "Statistics",      prerequisites=["spm_01"]),
    ConceptNode("spm_06", "Geometry",        prerequisites=[]),
    ConceptNode("spm_07", "Differentiation", prerequisites=["spm_03"]),
    ConceptNode("spm_08", "Integration",     prerequisites=["spm_07"]),
    ConceptNode("spm_09", "Probability",     prerequisites=["spm_05"]),
]:
    _GRAPH.register_node(_node)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Cognitive Decay Matrix API",
    description="DAG-based knowledge state engine with Ebbinghaus memory decay.",
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# DB ↔ Engine bridge helpers
# ---------------------------------------------------------------------------

def _hydrate_engine(student_id: str, db: Session) -> StudentEngine:
    """
    Build a StudentEngine pre-loaded with all states for this student from DB.
    Called at the start of every request so the HTTP layer stays stateless.
    """
    eng = StudentEngine(student_id=student_id, graph=_GRAPH)
    rows = (
        db.query(DBStudentState)
        .filter(DBStudentState.student_id == student_id)
        .all()
    )
    for row in rows:
        eng._states[row.node_id] = StudentState(
            node_id=row.node_id,
            current_weight=row.current_weight,
            last_updated=row.last_updated,
        )
    return eng


def _persist_state(student_id: str, node_id: str, eng: StudentEngine, db: Session) -> None:
    """
    Upsert the engine's in-memory state for (student_id, node_id) back to SQLite.
    """
    state = eng._states.get(node_id)
    if state is None:
        return
    row = (
        db.query(DBStudentState)
        .filter_by(student_id=student_id, node_id=node_id)
        .first()
    )
    if row:
        row.current_weight = state.current_weight
        row.last_updated   = state.last_updated
    else:
        db.add(DBStudentState(
            student_id=student_id,
            node_id=node_id,
            current_weight=state.current_weight,
            last_updated=state.last_updated,
        ))
    db.commit()


# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------

class AttemptRequest(BaseModel):
    student_id: str = Field(..., example="student_maya")
    node_id:    str = Field(..., example="spm_01")
    score:      float = Field(..., ge=0.0, le=1.0, example=1.0,
                              description="Normalised score: 0.0 = 0%, 1.0 = 100%")


class AttemptResponse(BaseModel):
    student_id:    str
    node_id:       str
    score:         float
    weight_before: float   # live (decayed) weight just before the attempt
    weight_after:  float   # stored weight after the update
    timestamp:     float


class PrereqStatus(BaseModel):
    node_id:        str
    name:           str
    live_weight:    float   # actual decayed value at query time
    live_weight_pct: float  # same value as a percentage
    status:         str     # "COLLAPSED" | "OK"


class DiagnosisResponse(BaseModel):
    student_id:           str
    failed_node_id:       str
    failed_node_name:     str
    timestamp:            float
    prerequisites:        List[PrereqStatus]
    collapsed_foundations: List[str]  # list of node_ids that are below threshold
    summary:              str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/record_attempt", response_model=AttemptResponse, status_code=200)
def record_attempt(req: AttemptRequest, db: Session = Depends(get_db)):
    """
    Record a student's assessment attempt on a concept node.

    - Loads existing state from SQLite.
    - Calls `StudentEngine.record_attempt()` (core CDM math, untouched).
    - Persists the updated weight back to SQLite.
    - Returns before/after weights so the caller can see the delta.
    """
    # Validate node exists in the static graph
    try:
        _GRAPH.get_node(req.node_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Node '{req.node_id}' is not registered in the Knowledge Graph.",
        )

    ts  = time.time()
    eng = _hydrate_engine(req.student_id, db)

    # Snapshot weight before the attempt (already decayed to now)
    weight_before = eng._live_weight(req.node_id, ts)

    # ── Core engine call (dag_engine.py — untouched) ──
    eng.record_attempt(req.node_id, req.score, ts)

    weight_after = eng._states[req.node_id].current_weight

    # Persist new state to SQLite
    _persist_state(req.student_id, req.node_id, eng, db)

    return AttemptResponse(
        student_id=req.student_id,
        node_id=req.node_id,
        score=req.score,
        weight_before=round(weight_before, 4),
        weight_after=round(weight_after, 4),
        timestamp=ts,
    )


@app.get("/diagnose/{student_id}/{node_id}", response_model=DiagnosisResponse)
def diagnose(student_id: str, node_id: str, db: Session = Depends(get_db)):
    """
    Trigger the CDM backpropagation diagnosis for a failed node.

    Performs a reverse DFS over the DAG, computes live (Ebbinghaus-decayed)
    weights for all transitive prerequisites from SQLite, and returns a
    structured JSON report identifying any collapsed foundations.
    """
    try:
        failed_node = _GRAPH.get_node(node_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Node '{node_id}' is not registered in the Knowledge Graph.",
        )

    ts  = time.time()
    eng = _hydrate_engine(student_id, db)

    # ── Reverse DFS using dag_engine's graph interface ──
    # (same algorithm as StudentEngine.diagnose_failure, structured output)
    visited: dict[str, float] = {}

    def _dfs(nid: str) -> None:
        if nid in visited:
            return
        for prereq_node in _GRAPH.get_prerequisites(nid):
            pid = prereq_node.node_id
            if pid not in visited:
                # eng._live_weight calls compute_decayed_weight from decay_math.py
                visited[pid] = eng._live_weight(pid, ts)
                _dfs(pid)

    _dfs(node_id)

    # Build structured prerequisite list, worst first
    prereq_list: List[PrereqStatus] = []
    collapsed:   List[str]          = []

    for pid, weight in sorted(visited.items(), key=lambda kv: kv[1]):
        name   = _GRAPH.get_node(pid).name
        status = "COLLAPSED" if weight < DECAY_THRESHOLD else "OK"
        prereq_list.append(PrereqStatus(
            node_id=pid,
            name=name,
            live_weight=round(weight, 4),
            live_weight_pct=round(weight * 100, 1),
            status=status,
        ))
        if weight < DECAY_THRESHOLD:
            collapsed.append(pid)

    # Human-readable summary
    if collapsed:
        worst      = min(visited, key=visited.get)
        worst_name = _GRAPH.get_node(worst).name
        summary = (
            f"Failure at '{failed_node.name}' is most likely caused by memory decay "
            f"in prerequisite '{worst_name}' (live weight: {visited[worst]*100:.1f}%). "
            f"Re-teach collapsed foundations before re-attempting this node."
        )
    elif not visited:
        summary = (
            f"'{failed_node.name}' has no prerequisites. "
            "Failure is likely due to first exposure or intrinsic topic difficulty."
        )
    else:
        summary = (
            f"All prerequisites for '{failed_node.name}' are above the "
            f"{DECAY_THRESHOLD*100:.0f}% threshold. Failure is likely due to "
            "intrinsic difficulty of this specific topic."
        )

    return DiagnosisResponse(
        student_id=student_id,
        failed_node_id=node_id,
        failed_node_name=failed_node.name,
        timestamp=ts,
        prerequisites=prereq_list,
        collapsed_foundations=collapsed,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "nodes_registered": len(_GRAPH.get_all_nodes()),
        "decay_threshold": DECAY_THRESHOLD,
    }


# ===========================================================================
# ADAPTIVE TUTOR ENDPOINTS
# ===========================================================================

# ---------------------------------------------------------------------------
# New Pydantic schemas
# ---------------------------------------------------------------------------

class JourneyNode(BaseModel):
    node_id: str
    name: str
    status: str           # COLLAPSED | NEEDS_REVIEW | LEARNING | MASTERED | LOCKED
    live_weight: float
    live_weight_pct: float
    prereq_ids: List[str]
    is_next: bool
    is_unlocked: bool


class JourneyResponse(BaseModel):
    student_id: str
    nodes: List[JourneyNode]
    next_node_id: Optional[str]


class LessonResponse(BaseModel):
    student_id: str
    node_id: str
    node_name: str
    lesson_markdown: str


class QuestionResponse(BaseModel):
    student_id: str
    node_id: str
    node_name: str
    question: str
    hint: str


class AnswerSubmission(BaseModel):
    student_id: str = Field(..., example="student_maya")
    node_id: str    = Field(..., example="spm_01")
    question: str   = Field(..., example="Solve: 2x + 3 = 7")
    answer: str     = Field(..., example="x = 2")


class AnswerResult(BaseModel):
    student_id: str
    node_id: str
    score: float
    is_correct: bool
    feedback: str
    explanation: str
    weight_before: float
    weight_after: float
    next_node_id: Optional[str]
    collapsed_prereqs: List[str]


# ---------------------------------------------------------------------------
# Journey endpoint
# ---------------------------------------------------------------------------

@app.get("/journey/{student_id}", response_model=JourneyResponse)
def get_student_journey(student_id: str, db: Session = Depends(get_db)):
    """
    Return the full knowledge journey for a student — all nodes ordered
    topologically with live mastery weights and recommended next step.
    """
    ts  = time.time()
    eng = _hydrate_engine(student_id, db)
    items = get_journey(student_id, _GRAPH, eng, ts)
    next_id = get_next_node(student_id, _GRAPH, eng, ts)

    return JourneyResponse(
        student_id=student_id,
        nodes=[JourneyNode(**item.to_dict()) for item in items],
        next_node_id=next_id,
    )


# ---------------------------------------------------------------------------
# Lesson endpoint
# ---------------------------------------------------------------------------

@app.get("/lesson/{student_id}/{node_id}", response_model=LessonResponse)
def get_lesson(student_id: str, node_id: str, db: Session = Depends(get_db)):
    """
    Generate (via local Ollama LLM) a teaching lesson for a concept node.
    Uses the student's prerequisite nodes as context.
    Prefetches a difficulty-adapted question using the student's real live weight.
    """
    try:
        node = _GRAPH.get_node(node_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found.")

    prereq_names = [_GRAPH.get_node(pid).name for pid in node.prerequisites if pid in _GRAPH.get_all_nodes()]

    try:
        lesson_md = generate_lesson(node.name, prereq_names)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # Prefetch a question at the student's actual mastery level (not hardcoded 0.0)
    ts = time.time()
    eng = _hydrate_engine(student_id, db)
    mastery = eng._live_weight(node_id, ts)
    prefetch_question(node.name, mastery_level=mastery)

    return LessonResponse(
        student_id=student_id,
        node_id=node_id,
        node_name=node.name,
        lesson_markdown=lesson_md,
    )


# ---------------------------------------------------------------------------
# Question endpoint
# ---------------------------------------------------------------------------

@app.get("/question/{student_id}/{node_id}", response_model=QuestionResponse)
def get_question(student_id: str, node_id: str, db: Session = Depends(get_db)):
    """
    Generate a practice question for a concept, difficulty-adapted to the
    student's current mastery weight.
    """
    try:
        node = _GRAPH.get_node(node_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found.")

    ts  = time.time()
    eng = _hydrate_engine(student_id, db)
    mastery = eng._live_weight(node_id, ts)

    try:
        q = generate_question(node.name, mastery)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return QuestionResponse(
        student_id=student_id,
        node_id=node_id,
        node_name=node.name,
        question=q.get("question", ""),
        hint=q.get("hint", ""),
    )


# ---------------------------------------------------------------------------
# Streaming answer evaluation (SSE) — live thinking/writing tokens
# ---------------------------------------------------------------------------

@app.post("/submit_answer_stream")
def submit_answer_stream_ep(req: AnswerSubmission):
    """
    SSE version of submit_answer.
    Streams events: checking → thinking_start → thinking tokens
    → writing tokens → done/correct → cdm_update.

    Uses its own DB session (not Depends) so the session stays alive
    for the full duration of the stream.
    """
    try:
        node = _GRAPH.get_node(req.node_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Node '{req.node_id}' not found.")

    ts = time.time()

    def event_stream():
        db = SessionLocal()
        try:
            eng = _hydrate_engine(req.student_id, db)
            weight_before = eng._live_weight(req.node_id, ts)

            eval_result: dict = {}
            score: float = 0.5

            for event in evaluate_answer_stream(node.name, req.question, req.answer):
                phase = event.get("phase")
                # Capture result from either "correct" or "done"
                if phase in ("correct", "done"):
                    eval_result = event.get("result", {})
                    score = eval_result.get("score", 0.5)
                yield f"data: {_api_json.dumps(event)}\n\n"

            # CDM update after streaming finishes
            eng.record_attempt(req.node_id, score, ts)
            weight_after = eng._states[req.node_id].current_weight
            _persist_state(req.student_id, req.node_id, eng, db)

            if score < 0.70:
                invalidate_question(node.name)

            collapsed: List[str] = []
            if score < 0.70:
                visited: dict = {}
                def _dfs(nid: str) -> None:
                    for prereq_node in _GRAPH.get_prerequisites(nid):
                        pid = prereq_node.node_id
                        if pid not in visited:
                            visited[pid] = eng._live_weight(pid, ts)
                            _dfs(pid)
                _dfs(req.node_id)
                collapsed = [pid for pid, w in visited.items() if w < DECAY_THRESHOLD]

            next_id = get_next_node(req.student_id, _GRAPH, eng, ts)
            is_correct = eval_result.get("is_correct", score >= 0.70)

            cdm_event = {
                "phase": "cdm_update",
                "student_id": req.student_id,
                "node_id": req.node_id,
                "score": round(score, 4),
                "is_correct": is_correct,
                "feedback": eval_result.get("feedback", ""),
                "explanation": eval_result.get("explanation", ""),
                "weight_before": round(weight_before, 4),
                "weight_after": round(weight_after, 4),
                "next_node_id": next_id,
                "collapsed_prereqs": collapsed,
            }
            yield f"data: {_api_json.dumps(cdm_event)}\n\n"

        except Exception as exc:
            import logging
            logging.getLogger(__name__).error("Stream error: %s", exc)
            yield f"data: {_api_json.dumps({'phase': 'error', 'message': str(exc)})}\n\n"
        finally:
            db.close()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Debug: Time Warp
# ---------------------------------------------------------------------------

class TimeWarpRequest(BaseModel):
    student_id: str = Field(..., example="student_maya")
    days_to_skip: int = Field(..., ge=1, le=3650, example=30)

@app.post("/debug/time_warp")
def debug_time_warp(req: TimeWarpRequest, db: Session = Depends(get_db)):
    """
    Subtract `days_to_skip` from every last_updated timestamp for the student,
    simulating the passage of time so Ebbinghaus decay takes effect.
    """
    seconds = req.days_to_skip * 86400
    rows = (
        db.query(DBStudentState)
        .filter(DBStudentState.student_id == req.student_id)
        .all()
    )
    if not rows:
        return {"warped": 0, "message": "No records found for this student."}
    for row in rows:
        row.last_updated -= seconds
    db.commit()
    return {
        "student_id": req.student_id,
        "days_skipped": req.days_to_skip,
        "records_warped": len(rows),
        "nodes": [r.node_id for r in rows],
    }


# ---------------------------------------------------------------------------
# Next node shortcut
# ---------------------------------------------------------------------------

@app.get("/next/{student_id}")
def get_next(student_id: str, db: Session = Depends(get_db)):
    """Return just the next recommended node_id for a student."""
    ts  = time.time()
    eng = _hydrate_engine(student_id, db)
    next_id = get_next_node(student_id, _GRAPH, eng, ts)
    if next_id is None:
        return {"next_node_id": None, "message": "All nodes mastered!"}
    node = _GRAPH.get_node(next_id)
    return {"next_node_id": next_id, "next_node_name": node.name}


# ---------------------------------------------------------------------------
# Serve dashboard (must be last — catches all unmatched paths)
# ---------------------------------------------------------------------------

_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/app", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")

@app.get("/", include_in_schema=False)
def root():
    index_path = os.path.join(_FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "CDM API running. Dashboard not found — ensure frontend/index.html exists."}
