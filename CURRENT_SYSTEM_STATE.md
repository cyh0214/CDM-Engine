# CURRENT_SYSTEM_STATE.md
## Cognitive Decay Matrix (CDM) — Full Technical Audit
**Generated:** 2026-03-02 | **Stack:** Python 3 · FastAPI · SQLite · Ollama (qwen3:4b) · Vanilla JS

> **Note on model name:** The user brief references "Llama 3.2". The system actually runs **qwen3:4b** via Ollama. This is the model in production. All LLM analysis below reflects qwen3:4b behaviour.

---

## 1. Math & DAG Engine (`decay_math.py`, `dag_engine.py`, `models.py`)

### 1.1 Ebbinghaus Decay Formula

**File:** `decay_math.py:8–24`

```
W_current = W_initial × e^(−λ × Δt)
```

| Symbol | Value | Meaning |
|--------|-------|---------|
| `λ` (DECAY_LAMBDA) | **0.05 / day** | Forgetting rate constant |
| `Δt` | `(ts_now − ts_last) / 86400.0` | Elapsed days (UNIX seconds → days) |
| `W_initial` | stored `current_weight` | Weight at time of last interaction |
| `W_current` | return value, clamped `[0.0, 1.0]` | Live decayed weight |

**Concrete decay values at λ = 0.05:**
| Days elapsed | Decay factor | 1.0 → |
|---|---|---|
| 7 days | e^(−0.35) = 0.705 | 70.5% |
| 14 days | e^(−0.70) = 0.497 | 49.7% |
| 30 days | e^(−1.50) = 0.223 | **22.3%** ← demo value |
| 60 days | e^(−3.00) = 0.050 | 5.0% |

**Implementation:** `compute_decayed_weight(initial_weight, elapsed_days)` in `decay_math.py`. Called exclusively via `StudentEngine._live_weight()` in `dag_engine.py:67–73`. The stored `current_weight` is **never decayed in-place** — decay is always computed on-the-fly at query time using `last_updated` timestamp.

---

### 1.2 Weight Update Rule (on `record_attempt`)

**File:** `dag_engine.py:88–124`

**Step 1:** Apply decay to the stored weight first:
```python
days_since = elapsed_days_between(state.last_updated, timestamp)
decayed = compute_decayed_weight(state.current_weight, days_since)
```

**Step 2 — Success path** (`score >= SUCCESS_THRESHOLD = 0.70`):
```python
new_weight = decayed + (score * LEARNING_RATE * (1.0 - decayed))
```
- Custom variable `LEARNING_RATE = 0.35` ensures knowledge climbs gradually, preventing instant mastery.
- At `decayed=0.0, score=1.0` → `new_weight = 0.350` (takes ~4 attempts to cross MASTERY_THRESHOLD=0.75)
- At `decayed=0.35, score=1.0` → `new_weight = 0.35 + 0.2275 = 0.5775`
- At `decayed=0.8, score=1.0` → `new_weight = 0.8 + 0.07 = 0.87`

**Step 2 — Failure path** (`score < 0.70`):
```python
penalty = (SUCCESS_THRESHOLD − score) / SUCCESS_THRESHOLD
new_weight = decayed × (1.0 − penalty × 0.5)
```
- At `decayed=0.8, score=0.0` → `penalty=1.0` → `new_weight = 0.8 × 0.5 = 0.40`
- At `decayed=0.8, score=0.35` → `penalty=0.5` → `new_weight = 0.8 × 0.75 = 0.60`

Final result is clamped to `[0.0, 1.0]`.

---

### 1.3 Threshold Constants

| Constant | Value | Location | Meaning |
|---|---|---|---|
| `DECAY_THRESHOLD` | **0.60** | `dag_engine.py:9` | Below this → node classified as COLLAPSED FOUNDATION |
| `SUCCESS_THRESHOLD` | **0.70** | `dag_engine.py:12` | Below this → failure path; above → success path |
| `MASTERY_THRESHOLD` | **0.75** | `knowledge_journey.py:19` | Above this → status = MASTERED |

**A "Collapsed Foundation"** is any node whose `_live_weight(node_id, now) < 0.60`. This triggers: (1) diagnosis report, (2) sidebar status = COLLAPSED (red), (3) `get_next_node()` returns that node as Priority 1.

---

### 1.4 Knowledge DAG — All 9 Nodes and Dependency Vectors

**File:** `api.py:43–54`

```
spm_06: Geometry        (no prerequisites)
spm_01: Algebra         (no prerequisites)
    └─► spm_02: Quadratics
            └─► spm_03: Calculus
                    └─► spm_07: Differentiation
                            └─► spm_08: Integration
    └─► spm_04: Trigonometry
    └─► spm_05: Statistics
            └─► spm_09: Probability
```

**Full dependency table:**

| node_id | Name | prerequisites |
|---|---|---|
| spm_01 | Algebra | `[]` |
| spm_02 | Quadratics | `["spm_01"]` |
| spm_03 | Calculus | `["spm_02"]` |
| spm_04 | Trigonometry | `["spm_01"]` |
| spm_05 | Statistics | `["spm_01"]` |
| spm_06 | Geometry | `[]` |
| spm_07 | Differentiation | `["spm_03"]` |
| spm_08 | Integration | `["spm_07"]` |
| spm_09 | Probability | `["spm_05"]` |

**Unlock condition:** A node is `is_unlocked = True` only if **all** its direct prerequisites have `_live_weight >= DECAY_THRESHOLD (0.60)`. If Algebra decays to 22% after 30 days, Quadratics, Trigonometry, and Statistics all become LOCKED simultaneously.

---

### 1.5 Reverse DFS Diagnosis Algorithm

**File:** `dag_engine.py:115–178`, mirrored in `api.py:230–242`

On failure (`score < 0.70`), a reverse DFS traverses all transitive prerequisites of the failed node, computing live weights. Any node with `live_weight < 0.60` is added to `collapsed_prereqs[]` in the API response.

```
Failed: Calculus (spm_03)
DFS order:
  Quadratics (spm_02) → live_weight checked
  Algebra (spm_01)    → live_weight checked (transitive)
Result: if Algebra < 0.60 → collapsed_prereqs = ["spm_01"]
```

---

## 2. LLM Orchestration (`tutor_engine.py`)

### 2.1 Model Configuration

| Parameter | Value |
|---|---|
| Model | `qwen3:4b` (Ollama local) |
| Endpoint | `http://localhost:11434/api/chat` |
| `think` | `True` (reasoning routed to separate `thinking` field) |
| `temperature` | `0.3` |
| `stream` | `True` (always; keeps HTTP connection alive during thinking phase) |
| Timeout | `300.0s` per call |

**Why streaming is mandatory:** qwen3:4b enters a thinking phase (15–120s) before writing its answer. With `stream=False`, Ollama buffers the entire response including thinking before sending, causing ReadTimeout. With `stream=True`, chunks arrive continuously. The `think=True` parameter routes reasoning tokens to `chunk["message"]["thinking"]` and answer tokens to `chunk["message"]["content"]` — we discard thinking and collect only content.

---

### 2.2 System Prompts

**Lesson / Question generation** (`_TUTOR_CONTEXT`):
```
"You are a concise SPM Mathematics tutor for secondary school students.
Give short, clear answers only. Use $...$ for inline math and $$...$$ for
display math. Never add unnecessary words."
```

**Evaluation** (`_EVAL_CONTEXT`):
```
"You are grading a student's maths answer. The expected answer is already
given — do NOT re-solve the problem yourself. Your job: (1) decide if the
student is correct, (2) give one sentence of targeted feedback about their
specific mistake, (3) show the step-by-step working. Be encouraging and
specific. Use $...$ for inline math."
```

---

### 2.3 Lesson Prompt Structure

**File:** `tutor_engine.py:140–157`

```python
prompt = (
    f"{prereq_str}"
    f"Teach **{concept_name}** for SPM Mathematics.\n\n"
    "Structure your lesson in markdown with these sections:\n"
    "## Definition\n## Key Formula\n## Worked Example\n"
    "## Common Mistake\n## Remember\n\n"
    "Keep under 350 words. Use $...$ for inline math, $$...$$ for display equations."
)
```

`prereq_str` is either `"The student already knows: Algebra, Quadratics. "` or `"This is the student's first maths topic. "`.

---

### 2.4 Lesson Cache (`_lesson_cache`)

**File:** `tutor_engine.py:38, 92–112`

```python
_lesson_cache: dict[str, str] = {}
# Key: concept_name + "|" + ",".join(prereq_names)
# Value: full markdown lesson string
```

- **Scope:** Process-level in-memory dict. Shared across all students.
- **Invalidation:** Never. Cache lives for the entire server process lifetime.
- **Effect:** First call per concept takes ~12–20s. All subsequent calls for any student on the same concept are **instant** (0.01–0.02s).
- **Cache miss path:** `generate_lesson()` → `_chat(prompt)` → Ollama HTTP stream → collect content tokens → store in `_lesson_cache[key]`.

---

### 2.5 Question Cache and Prefetch Mechanism

**File:** `tutor_engine.py:53, 191–251`

```python
_question_cache: dict[str, dict] = {}
# Key: f"{concept_name}:{bucket}" where bucket is "basic", "standard", or "advanced"
# Value: {question: str, hint: str, answer: str, _stale?: bool}
```

**Prefetch flow:**
1. `GET /lesson/{student_id}/{node_id}` responds with lesson content.
2. Before returning, calls `prefetch_question(node.name, mastery_level=mastery)` — **file:** `api.py:415`.
3. `prefetch_question()` checks if a fresh (non-stale) entry exists in `_question_cache` for the appropriate difficulty bucket. If not, submits `_safe_prefetch()` to a `ThreadPoolExecutor(max_workers=2)`.
4. `_safe_prefetch()` calls `generate_question()` which calls Ollama and caches the result under the specific bucket.
5. By the time the student finishes reading the lesson (~12–30s of reading time), the question is already cached.
6. `GET /question/...` returns the cached entry instantly.

**Stale mechanism (after wrong answer):**
- `submit_answer_stream_ep` calls `invalidate_question(node.name)` when `score < 0.70`.
- `invalidate_question()` sets `_question_cache[key]["_stale"] = True` for all difficulty buckets of that concept.
- The expected `answer` value is **preserved** in the stale entry so `evaluate_answer_stream()` can still compare it.
- Next call to `generate_question()` sees `_stale=True` → bypasses cache → generates fresh question → overwrites entry without `_stale` flag.

---

### 2.6 Answer Evaluation — Score Extraction

**File:** `tutor_engine.py:218–260`

**Correct path (instant, no LLM):**
```python
s_norm = _normalise(student_answer)   # lowercase, strip whitespace/punctuation
e_norm = _normalise(expected_answer)
is_correct = (s_norm == e_norm) or _numbers_match(s_norm, e_norm)
```
`_numbers_match()` extracts the first numeric value from each string via regex `r"-?[\d]+(?:\.[\d]+)?"` and compares with tolerance `< 1e-6`. Returns `score=1.0` instantly.

**Wrong path (LLM, ~5–15s):**
Prompt sent to Ollama with expected answer already included:
```
Question: {question}
Expected answer: {expected}
Student wrote: {student_answer}

Reply in this exact format — four lines only:
VERDICT: correct OR incorrect
SCORE: 0-10
FEEDBACK: <one sentence>
SOLUTION: <step-by-step working>
```

Score extraction in `_parse_evaluation()`:
```python
score = max(0.0, min(10.0, float(re.search(r"[\d.]+", score_str).group()))) / 10.0
```
Regex finds first number in the SCORE line, divides by 10 to normalise to `[0.0, 1.0]`. If score ≥ 0.9 but student was confirmed wrong (we called `_smart_evaluate`), it is clamped to 0.5.

`is_correct` from parsed response: `("incorrect" not in verdict) and ("correct" in verdict or score >= 0.7)`.

---

### 2.7 Streaming Evaluation Events (`evaluate_answer_stream`)

**File:** `tutor_engine.py:282–355`

| Event Phase | Trigger | Payload |
|---|---|---|
| `checking` | Always first | `{message: "Comparing your answer…"}` |
| `correct` | String/number match | `{result: {score:1.0, is_correct:true, feedback, explanation}}` |
| `thinking_start` | Wrong answer detected | `{message: "Answer doesn't match (expected: X). Analysing…"}` |
| `thinking` | Each thinking token from Ollama | `{token: str}` — model's internal reasoning |
| `writing` | Each content token from Ollama | `{token: str}` — model writing the answer |
| `done` | LLM stream complete | `{result: {score, is_correct, feedback, explanation}}` |
| `error` | Exception | `{message: str}` |

After the generator exhausts, `submit_answer_stream_ep` emits one final event:

| Event Phase | Trigger | Payload |
|---|---|---|
| `cdm_update` | Post-stream CDM write | Full result: score, weights, next_node_id, collapsed_prereqs |

---

## 3. State Management & API (`api.py`, `database.py`)

### 3.1 Database Schema

**File:** `database.py:26–42`

```sql
CREATE TABLE student_states (
    student_id     TEXT NOT NULL,
    node_id        TEXT NOT NULL,
    current_weight REAL NOT NULL,   -- stored weight (NOT decayed)
    last_updated   REAL NOT NULL,   -- UNIX timestamp (float, seconds)
    PRIMARY KEY (student_id, node_id)
);
```

**Critical design:** `current_weight` is the weight **at the moment of last interaction**. It is never decayed in-place. The live decayed weight is always computed at request time: `W_live = current_weight × e^(−0.05 × (now − last_updated) / 86400)`.

### 3.2 Request Lifecycle (Stateless HTTP Pattern)

Every mutating request follows this pattern:
1. `_hydrate_engine(student_id, db)` — loads all `DBStudentState` rows for the student into a fresh `StudentEngine` in-memory.
2. Engine performs computation (decay, update, diagnosis).
3. `_persist_state(student_id, node_id, eng, db)` — upserts the single modified row back to SQLite.
4. Engine is discarded; next request starts fresh from DB.

---

### 3.3 All Active Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | None | Returns node count + decay threshold |
| `POST` | `/record_attempt` | None | Raw CDM weight update (no LLM) |
| `GET` | `/diagnose/{student_id}/{node_id}` | None | DFS diagnosis report |
| `GET` | `/journey/{student_id}` | None | Full topological journey with live weights |
| `GET` | `/lesson/{student_id}/{node_id}` | None | LLM-generated lesson (cached) + triggers question prefetch |
| `GET` | `/question/{student_id}/{node_id}` | None | Practice question (mastery-adapted, cached) |
| `POST` | `/submit_answer_stream` | None | SSE streaming evaluation + CDM update |
| `GET` | `/next/{student_id}` | None | Returns recommended next node_id |
| `POST` | `/debug/time_warp` | None | Subtract days from all timestamps (demo tool) |
| `GET` | `/` | None | Serves `frontend/index.html` |
| `GET` | `/app/*` | None | Static files from `frontend/` directory |

---

### 3.4 Time Warp Mechanism — Exact Implementation

**File:** `api.py:624–646`

```python
seconds = req.days_to_skip * 86400          # e.g. 30 × 86400 = 2,592,000
rows = db.query(DBStudentState)
         .filter(DBStudentState.student_id == req.student_id)
         .all()
for row in rows:
    row.last_updated -= seconds             # shift timestamp backwards in time
db.commit()
```

**Mathematical effect on decay:**

Before warp (student mastered Algebra today, `last_updated = now`):
```
Δt = (now − now) / 86400 = 0 days
W_live = 1.0 × e^(0) = 1.0   (100%)
```

After warp of 30 days (`last_updated = now − 2,592,000`):
```
Δt = (now − (now − 2592000)) / 86400 = 30 days
W_live = 1.0 × e^(−0.05 × 30) = 1.0 × e^(−1.5) = 0.223   (22.3%)
```

`0.223 < DECAY_THRESHOLD (0.60)` → node status flips to **COLLAPSED FOUNDATION**.

The warp is purely a timestamp manipulation. No weights are changed. The decay formula itself produces the new live weight on the next GET /journey request.

---

## 4. Frontend Architecture (`frontend/index.html`)

### 4.1 Single-File Architecture

The entire frontend is one HTML file (`frontend/index.html`, ~700 lines). No build step, no framework. Dependencies:
- **KaTeX 0.16.9** (CDN) — math rendering for `$...$` and `$$...$$`
- **Vanilla JS** — all state, fetch, SSE handling, and DOM manipulation
- **CSS custom properties** — dark theme via `:root` variables

### 4.2 Application State Object

```javascript
const state = {
    studentId: null,          // lowercased, spaces→underscore version of input
    currentNodeId: null,      // active node being studied
    currentNodeLocked: false, // whether current node is locked
    currentQuestion: null,    // question text (needed for /submit_answer_stream body)
    journeyData: null,        // last /journey response
};
```

### 4.3 Collapsed Foundation UI Behaviour

When `POST /submit_answer_stream` emits the `cdm_update` event with `collapsed_prereqs: ["spm_01"]`:

**File:** `frontend/index.html:545–545` (submitAnswer function)

```javascript
if (result.collapsed_prereqs?.length) {
    const names = result.collapsed_prereqs.map(id => {
        const n = state.journeyData?.nodes.find(x => x.node_id === id);
        return n ? n.name : id;
    }).join(', ');
    cw.style.display = 'block';
    cw.textContent = `⚠️ Weak foundations: ${names}. These are now prioritised in your journey.`;
}
```

This shows `#collapsed-warning` — an orange-bordered banner listing the collapsed prerequisite names.

**The system does NOT force-redirect the student.** It uses a soft nudge:
1. Orange warning banner appears inline in the feedback panel.
2. `loadJourney(false)` silently refreshes the sidebar — the collapsed node now shows status `COLLAPSED` with a red badge and 0% bar.
3. The "Next" button text is driven by `result.next_node_id`, which comes from `get_next_node()`.

**`get_next_node()` Priority 1** (`knowledge_journey.py:146–150`):
```python
for attempted_id in list(engine._states.keys()):
    for prereq_node in graph.get_prerequisites(attempted_id):
        pid = prereq_node.node_id
        if engine._live_weight(pid, ts) < DECAY_THRESHOLD:
            return pid   # ← returns the collapsed prereq first
```

So after time warp, if Algebra is collapsed, `next_node_id = "spm_01"` is returned. The "Next" button becomes **"Continue practising: Algebra →"**, guiding (not forcing) the student back to the foundation.

### 4.4 Live AI Console

**File:** `frontend/index.html` — `#ai-console` div + `submitAnswer()` function

The SSE stream from `/submit_answer_stream` is consumed via `fetch` + `ReadableStream`:
```javascript
const reader = response.body.getReader();
const decoder = new TextDecoder();
```

Each `data: {...}` line is parsed and routed to the console:

| Phase | Visual output |
|---|---|
| `checking` | `🔍 Comparing to expected answer…` (grey status line) |
| `thinking_start` | `🧠 Answer doesn't match. Analysing mistake…` (grey status line) |
| `thinking` tokens | Appended to `<span class="cl-thinking">` (dark, `#334155`, monospace) |
| `writing` tokens | Appended to `<span class="cl-writing">` (purple, `#a5b4fc`) |
| `cdm_update` | Console stays; feedback panel appears below with KaTeX-rendered solution |

### 4.5 Math Rendering

KaTeX `auto-render` is loaded deferred. After any HTML injection, `renderMath(el)` is called:
```javascript
renderMathInElement(el, {
    delimiters: [
        {left:'$$', right:'$$', display:true},
        {left:'$',  right:'$',  display:false},
        {left:'\\(', right:'\\)', display:false},
        {left:'\\[', right:'\\]', display:true},
    ],
    throwOnError: false,
});
```
Called after: lesson load, question load, feedback panel population.

---

## 5. File Inventory

| File | Lines | Role |
|---|---|---|
| `models.py` | 28 | `ConceptNode` + `StudentState` dataclasses |
| `decay_math.py` | 31 | Ebbinghaus formula + timestamp utils |
| `dag_engine.py` | 179 | `KnowledgeGraph` + `StudentEngine` (weight updates, DFS diagnosis) |
| `knowledge_journey.py` | 187 | Journey ordering (Kahn's toposort) + `get_next_node()` priority logic |
| `database.py` | 57 | SQLAlchemy SQLite ORM, `SessionLocal`, `get_db` |
| `tutor_engine.py` | ~370 | LLM calls, caches, streaming, evaluation |
| `api.py` | ~680 | FastAPI endpoints, hydration/persistence bridge |
| `frontend/index.html` | ~720 | Single-file dashboard (CSS + HTML + JS) |
| `main.py` | — | Standalone simulation script (not used in HTTP server) |
| `cdm_state.db` | — | SQLite database (auto-created on first run) |

---

## 6. Known Limitations / Technical Debt

| # | Issue | Location | Impact |
|---|---|---|---|
| 1 | Lesson cache never evicts — server restart clears it | `_lesson_cache` in `tutor_engine.py` | Minor: ~15s regeneration after restart |
| 2 | `_question_cache` is shared per concept and difficulty bucket, not strictly per student | `tutor_engine.py:203` | Students at the exact same mastery bucket share questions |
| 3 | No authentication — any student_id string creates a new student | `api.py` | Demo-only system |
| 4 | `get_next_node()` Priority 1 returns first collapsed prereq of first attempted node, not worst | `knowledge_journey.py:146` | Sub-optimal ordering edge case |
| 5 | `diagnose_failure()` in `dag_engine.py` prints to stdout | `dag_engine.py:125-129` | Console noise in production |
