"""
knowledge_journey.py — CDM-aware study plan curator.

Decides what a student should study next by combining:
  - The DAG structure (prerequisite ordering)
  - Live (Ebbinghaus-decayed) mastery weights from StudentEngine
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from dag_engine import KnowledgeGraph, StudentEngine, DECAY_THRESHOLD, SUCCESS_THRESHOLD


# A node is considered "mastered" when its live weight exceeds this value.
MASTERY_THRESHOLD: float = 0.75


@dataclass
class JourneyItem:
    node_id: str
    name: str
    status: str          # COLLAPSED | NEEDS_REVIEW | LEARNING | MASTERED | LOCKED
    live_weight: float   # 0.0–1.0
    live_weight_pct: float
    prereq_ids: List[str]
    is_next: bool        # True for the single recommended next node
    is_unlocked: bool    # False if any prerequisite is below DECAY_THRESHOLD

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "status": self.status,
            "live_weight": round(self.live_weight, 4),
            "live_weight_pct": round(self.live_weight_pct, 1),
            "prereq_ids": self.prereq_ids,
            "is_next": self.is_next,
            "is_unlocked": self.is_unlocked,
        }


def _classify_status(live_weight: float, is_unlocked: bool, ever_attempted: bool) -> str:
    """Map a live weight + context to a human-readable status label."""
    if not is_unlocked:
        return "LOCKED"
    if not ever_attempted:
        return "LEARNING"          # unlocked but never touched
    if live_weight < DECAY_THRESHOLD:
        return "COLLAPSED"         # needs immediate reinforcement
    if live_weight < SUCCESS_THRESHOLD:
        return "NEEDS_REVIEW"
    if live_weight >= MASTERY_THRESHOLD:
        return "MASTERED"
    return "LEARNING"


def get_journey(
    student_id: str,
    graph: KnowledgeGraph,
    engine: StudentEngine,
    timestamp: Optional[float] = None,
) -> List[JourneyItem]:
    """
    Return all knowledge graph nodes as JourneyItems, ordered by:
      1. Topological depth (prerequisite nodes first)
      2. Alphabetical within the same depth

    The single best next node to study has `is_next = True`.
    """
    ts = timestamp or time.time()
    all_nodes = graph.get_all_nodes()

    # --- Topological sort (Kahn's algorithm) for display ordering ---
    in_degree: dict[str, int] = {nid: 0 for nid in all_nodes}
    for node in all_nodes.values():
        for pid in node.prerequisites:
            if pid in in_degree:
                in_degree[node.node_id] += 1

    queue = sorted([nid for nid, deg in in_degree.items() if deg == 0])
    topo_order: list[str] = []
    while queue:
        nid = queue.pop(0)
        topo_order.append(nid)
        for candidate in sorted(all_nodes.keys()):
            if nid in all_nodes[candidate].prerequisites:
                in_degree[candidate] -= 1
                if in_degree[candidate] == 0:
                    queue.append(candidate)
                    queue.sort()

    # --- Build JourneyItems ---
    next_node_id = get_next_node(student_id, graph, engine, ts)
    items: list[JourneyItem] = []

    for nid in topo_order:
        node = all_nodes[nid]
        live_w = engine._live_weight(nid, ts)
        ever_attempted = nid in engine._states

        # A node is unlocked if all its prerequisites have live weight >= DECAY_THRESHOLD
        is_unlocked = all(
            engine._live_weight(pid, ts) >= DECAY_THRESHOLD
            for pid in node.prerequisites
        ) if node.prerequisites else True

        status = _classify_status(live_w, is_unlocked, ever_attempted)

        items.append(JourneyItem(
            node_id=nid,
            name=node.name,
            status=status,
            live_weight=live_w,
            live_weight_pct=live_w * 100,
            prereq_ids=list(node.prerequisites),
            is_next=(nid == next_node_id),
            is_unlocked=is_unlocked,
        ))

    return items


def get_next_node(
    student_id: str,
    graph: KnowledgeGraph,
    engine: StudentEngine,
    timestamp: Optional[float] = None,
) -> Optional[str]:
    """
    Determine the single best node to study next.

    Priority order:
      1. Collapsed prerequisites (live_weight < DECAY_THRESHOLD) of any
         attempted node — fix foundations first.
      2. Unlocked nodes with the lowest live_weight (need most reinforcement).
      3. Unlocked nodes never attempted (in topological order).
    """
    ts = timestamp or time.time()
    all_nodes = graph.get_all_nodes()

    # --- Priority 1: collapsed prerequisites of nodes already attempted ---
    for attempted_id in list(engine._states.keys()):
        for prereq_node in graph.get_prerequisites(attempted_id):
            pid = prereq_node.node_id
            if engine._live_weight(pid, ts) < DECAY_THRESHOLD:
                return pid

    # Collect unlocked nodes
    unlocked_unattempted: list[tuple[str, float]] = []
    unlocked_attempted: list[tuple[str, float]] = []

    for nid, node in all_nodes.items():
        prereqs_ok = all(
            engine._live_weight(pid, ts) >= DECAY_THRESHOLD
            for pid in node.prerequisites
        ) if node.prerequisites else True

        if not prereqs_ok:
            continue

        live_w = engine._live_weight(nid, ts)

        if nid not in engine._states:
            unlocked_unattempted.append((nid, live_w))
        else:
            unlocked_attempted.append((nid, live_w))

    # --- Priority 2: lowest mastery among attempted unlocked nodes (but not mastered) ---
    needs_work = [(nid, w) for nid, w in unlocked_attempted if w < MASTERY_THRESHOLD]
    if needs_work:
        return min(needs_work, key=lambda x: x[1])[0]

    # --- Priority 3: first unattempted unlocked node (topological order) ---
    if unlocked_unattempted:
        # Sort by node_id to keep topological-ish ordering (spm_01, spm_02…)
        return sorted(unlocked_unattempted, key=lambda x: x[0])[0][0]

    # All nodes mastered — return the one with lowest weight for review
    if unlocked_attempted:
        return min(unlocked_attempted, key=lambda x: x[1])[0]

    return None
