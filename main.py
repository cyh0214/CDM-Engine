"""
main.py — Cognitive Decay Matrix (CDM) Simulation
===================================================
Scenario: SPM Mathematics (Malaysia)

Timeline
--------
  T-30 days : Student aces Algebra (100%).
  T-15 days : Student aces Quadratics (90%).
  T-0  days : Student attempts Calculus — FAILS (40%).

The CDM must trace the Calculus failure back to the decayed Algebra foundation.
"""

import time
import math
from models import ConceptNode
from decay_math import DECAY_LAMBDA, compute_decayed_weight
from dag_engine import KnowledgeGraph, StudentEngine, DECAY_THRESHOLD

# ---------------------------------------------------------------------------
# Time constants
# ---------------------------------------------------------------------------
SECONDS_PER_DAY = 86_400
NOW = 1_700_000_000.0          # arbitrary fixed "today" (UNIX timestamp)
T_MINUS_30 = NOW - 30 * SECONDS_PER_DAY
T_MINUS_15 = NOW - 15 * SECONDS_PER_DAY


def print_section(title: str) -> None:
    bar = "─" * 68
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


# ---------------------------------------------------------------------------
# 1. Build the Knowledge DAG
# ---------------------------------------------------------------------------
print_section("PHASE 1 — BUILD SPM MATHEMATICS KNOWLEDGE DAG")

graph = KnowledgeGraph()

algebra    = ConceptNode(node_id="spm_01", name="Algebra",    prerequisites=[])
quadratics = ConceptNode(node_id="spm_02", name="Quadratics", prerequisites=["spm_01"])
calculus   = ConceptNode(node_id="spm_03", name="Calculus",   prerequisites=["spm_02"])

for node in (algebra, quadratics, calculus):
    graph.register_node(node)
    print(f"  Registered → {node}")

# ---------------------------------------------------------------------------
# 2. Create the student engine
# ---------------------------------------------------------------------------
print_section("PHASE 2 — INITIALISE STUDENT ENGINE")
engine = StudentEngine(student_id="student_maya", graph=graph)
print(f"  Student 'student_maya' enrolled.")

# ---------------------------------------------------------------------------
# 3. Simulate past learning events
# ---------------------------------------------------------------------------
print_section("PHASE 3 — SIMULATE HISTORICAL LEARNING EVENTS")

print("\n  [T-30 days]  Maya sits the Algebra exam...")
engine.record_attempt("spm_01", score=1.00, timestamp=T_MINUS_30)

print("\n  [T-15 days]  Maya sits the Quadratics exam...")
engine.record_attempt("spm_02", score=0.90, timestamp=T_MINUS_15)

# ---------------------------------------------------------------------------
# 4. Show live decayed weights BEFORE the Calculus attempt
# ---------------------------------------------------------------------------
print_section("PHASE 4 — LIVE WEIGHT SNAPSHOT (at T=0, before Calculus attempt)")

nodes_to_check = [("spm_01", "Algebra"), ("spm_02", "Quadratics")]
for nid, name in nodes_to_check:
    state = engine._states[nid]
    days_elapsed = (NOW - state.last_updated) / SECONDS_PER_DAY
    live_w = compute_decayed_weight(state.current_weight, days_elapsed)
    expected = state.current_weight * math.exp(-DECAY_LAMBDA * days_elapsed)
    print(
        f"  [{nid}] {name:<14}  "
        f"stored_weight={state.current_weight:.4f}  "
        f"days_since={days_elapsed:.1f}  "
        f"live_weight={live_w:.4f}  ({live_w*100:.1f}%)  "
        f"{'⚠ BELOW THRESHOLD' if live_w < DECAY_THRESHOLD else '✓ OK'}"
    )

# ---------------------------------------------------------------------------
# 5. Student fails Calculus today
# ---------------------------------------------------------------------------
print_section("PHASE 5 — TODAY: Maya attempts Calculus and FAILS")

print("\n  [T=0]  Maya attempts Calculus (40% score — FAIL)...")
engine.record_attempt("spm_03", score=0.40, timestamp=NOW)

# ---------------------------------------------------------------------------
# 6. Run the CDM diagnosis
# ---------------------------------------------------------------------------
diagnosis = engine.diagnose_failure("spm_03", current_timestamp=NOW)

print()
for line in diagnosis:
    print(line)

# ---------------------------------------------------------------------------
# 7. Summary banner
# ---------------------------------------------------------------------------
print_section("CDM SYSTEM SUMMARY")
alg_state = engine._states["spm_01"]
alg_days  = (NOW - alg_state.last_updated) / SECONDS_PER_DAY
alg_live  = compute_decayed_weight(alg_state.current_weight, alg_days)

print(f"""
  STUDENT  : student_maya
  FAILED   : spm_03 — Calculus

  ROOT-CAUSE TRACE:
    ╔══════════════╗     ╔═══════════════╗     ╔══════════════╗
    ║  spm_01      ║ ──► ║  spm_02       ║ ──► ║  spm_03      ║
    ║  Algebra     ║     ║  Quadratics   ║     ║  Calculus    ║
    ║  {alg_live*100:5.1f}%      ║     ║  studied 15d  ║     ║  FAILED 40%  ║
    ╚══════════════╝     ╚═══════════════╝     ╚══════════════╝
         ↑
    COLLAPSED FOUNDATION (below {DECAY_THRESHOLD*100:.0f}% threshold)

  RECOMMENDATION: Re-teach Algebra before re-attempting Calculus.
  The Ebbinghaus model predicts Maya retained only {alg_live*100:.1f}% of her
  Algebra mastery after {alg_days:.0f} days without reinforcement.
""")

print("=" * 68)
print("  Cognitive Decay Matrix — simulation complete.")
print("=" * 68)
