from __future__ import annotations

from typing import Dict, List, Optional
from models import ConceptNode, StudentState
from decay_math import compute_decayed_weight, elapsed_days_between

# A node is considered a "collapsed foundation" if its decayed weight
# has fallen below this threshold.
DECAY_THRESHOLD: float = 0.60

# Minimum score on an attempt to consider it a success (weight boost).
SUCCESS_THRESHOLD: float = 0.70

# Fraction of the remaining gap closed per successful attempt.
# At 0.35: a student starting from 0.0 needs ~4 correct answers to reach
# the MASTERY_THRESHOLD of 0.75, preventing instant single-attempt mastery.
#   attempt 1: 0.00 + 1.0*0.35*(1-0.00) = 0.350
#   attempt 2: 0.35 + 1.0*0.35*(1-0.35) = 0.578
#   attempt 3: 0.58 + 1.0*0.35*(1-0.58) = 0.725
#   attempt 4: 0.73 + 1.0*0.35*(1-0.73) = 0.819  ← crosses 0.75 MASTERED
LEARNING_RATE: float = 0.35


class KnowledgeGraph:
    """
    Static DAG of concept nodes.
    Nodes must be registered before the engine can use them.
    """

    def __init__(self):
        self._nodes: Dict[str, ConceptNode] = {}

    def register_node(self, node: ConceptNode) -> None:
        """Add a ConceptNode to the graph."""
        self._nodes[node.node_id] = node

    def get_node(self, node_id: str) -> ConceptNode:
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' not found in graph.")
        return self._nodes[node_id]

    def get_all_nodes(self) -> Dict[str, ConceptNode]:
        return dict(self._nodes)

    def get_prerequisites(self, node_id: str) -> List[ConceptNode]:
        """Return the immediate prerequisite nodes for a given node."""
        node = self.get_node(node_id)
        return [self._nodes[pid] for pid in node.prerequisites if pid in self._nodes]


class StudentEngine:
    """
    Stateful engine that tracks and analyses one student's knowledge
    across the full KnowledgeGraph.
    """

    def __init__(self, student_id: str, graph: KnowledgeGraph):
        self.student_id = student_id
        self.graph = graph
        # node_id -> StudentState
        self._states: Dict[str, StudentState] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_init_state(self, node_id: str, timestamp: float) -> StudentState:
        if node_id not in self._states:
            self._states[node_id] = StudentState(
                node_id=node_id,
                current_weight=0.0,
                last_updated=timestamp,
            )
        return self._states[node_id]

    def _live_weight(self, node_id: str, current_timestamp: float) -> float:
        """Return the *decayed* weight for a node at the given timestamp."""
        if node_id not in self._states:
            return 0.0
        state = self._states[node_id]
        days = elapsed_days_between(state.last_updated, current_timestamp)
        return compute_decayed_weight(state.current_weight, days)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_attempt(self, node_id: str, score: float, timestamp: float) -> None:
        """
        Record an assessment attempt and update mastery weight.

        score: normalised [0.0, 1.0]   (1.0 == 100%)

        Weight update rule:
          - First apply decay to the *current* stored weight.
          - Then blend: new_weight = decayed_weight + score * (1 - decayed_weight)
            This means a perfect score on a well-known node barely moves it
            (already near 1.0), while it gives a large boost to fresh learning.
          - Clamp result to [0.0, 1.0].
        """
        state = self._get_or_init_state(node_id, timestamp)

        # 1. Apply time-decay to existing weight first
        days_since = elapsed_days_between(state.last_updated, timestamp)
        decayed = compute_decayed_weight(state.current_weight, days_since)

        # 2. Score-based update
        if score >= SUCCESS_THRESHOLD:
            # Success: weight climbs toward 1.0 at LEARNING_RATE pace.
            # Prevents instant mastery — requires multiple correct answers.
            new_weight = decayed + (score * LEARNING_RATE * (1.0 - decayed))
        else:
            # Failure: weight drops proportionally to how bad the score was
            penalty = (SUCCESS_THRESHOLD - score) / SUCCESS_THRESHOLD
            new_weight = decayed * (1.0 - penalty * 0.5)

        state.current_weight = max(0.0, min(1.0, new_weight))
        state.last_updated = timestamp

        print(
            f"  [RECORD] student={self.student_id!r}  node={node_id!r}  "
            f"score={score:.0%}  weight: {decayed:.4f} → {state.current_weight:.4f}"
        )

    def diagnose_failure(self, node_id: str, current_timestamp: float) -> List[str]:
        """
        Core diagnostic algorithm.

        When a student fails 'node_id', perform a reverse DFS over all
        transitive prerequisites, computing their live (decayed) weights.

        Returns a list of human-readable diagnostic lines sorted by weight
        (lowest first = most likely root cause).
        """
        print(f"\n{'='*68}")
        print(f"  COGNITIVE DECAY MATRIX — FAILURE DIAGNOSIS")
        print(f"  Student : {self.student_id}")
        print(f"  Failed  : {node_id}  ({self.graph.get_node(node_id).name})")
        print(f"{'='*68}")

        # ---- DFS to collect all transitive prerequisites ----
        visited: Dict[str, float] = {}   # node_id -> live_weight

        def _dfs(nid: str) -> None:
            if nid in visited:
                return
            prereqs = self.graph.get_prerequisites(nid)
            for prereq_node in prereqs:
                pid = prereq_node.node_id
                if pid not in visited:
                    live_w = self._live_weight(pid, current_timestamp)
                    visited[pid] = live_w
                    print(
                        f"  [DFS]  Checking prereq '{pid}' ({prereq_node.name})  "
                        f"→  live_weight = {live_w:.4f}  ({live_w*100:.1f}%)"
                    )
                    _dfs(pid)   # recurse deeper

        _dfs(node_id)

        # ---- Identify collapsed foundations ----
        report_lines: List[str] = []

        if not visited:
            report_lines.append(
                f"  NOTE: '{node_id}' has no prerequisites tracked. "
                "Failure may be due to first exposure or topic difficulty."
            )
        else:
            # Sort by weight ascending (worst first)
            sorted_prereqs = sorted(visited.items(), key=lambda kv: kv[1])

            for pid, weight in sorted_prereqs:
                prereq_name = self.graph.get_node(pid).name
                failed_name = self.graph.get_node(node_id).name
                status = "COLLAPSED FOUNDATION" if weight < DECAY_THRESHOLD else "OK"
                line = (
                    f"  ★ DIAGNOSIS: Failure at [{failed_name}] is likely due to "
                    f"decay in prerequisite [{prereq_name}] ('{pid}') "
                    f"which has dropped to {weight*100:.1f}% weight.  [{status}]"
                    if weight < DECAY_THRESHOLD
                    else
                    f"  ✓ Prereq [{prereq_name}] ('{pid}') is healthy at "
                    f"{weight*100:.1f}% weight.  [{status}]"
                )
                report_lines.append(line)

        return report_lines
