from dataclasses import dataclass, field
from typing import List


@dataclass
class ConceptNode:
    """Represents a knowledge concept in the DAG."""
    node_id: str
    name: str
    prerequisites: List[str] = field(default_factory=list)

    def __repr__(self):
        return f"ConceptNode(id={self.node_id!r}, name={self.name!r}, prereqs={self.prerequisites})"


@dataclass
class StudentState:
    """Tracks a student's mastery state for a single concept node."""
    node_id: str
    current_weight: float       # 0.0 (no mastery) to 1.0 (full mastery)
    last_updated: float         # UNIX timestamp of last interaction

    def __repr__(self):
        return (
            f"StudentState(node={self.node_id!r}, "
            f"weight={self.current_weight:.4f}, "
            f"last_updated={self.last_updated})"
        )
