import math

# Ebbinghaus decay constant: λ = 0.05 per day.
# A weight decays to ~22% after 30 days of no reinforcement.
DECAY_LAMBDA: float = 0.05


def compute_decayed_weight(initial_weight: float, elapsed_days: float) -> float:
    """
    Apply Ebbinghaus Forgetting Curve to a memory weight.

    Formula:  W_current = W_initial * e^(-λ * Δt)

    Args:
        initial_weight: Mastery weight at the time of last interaction [0.0, 1.0].
        elapsed_days:   Number of days since the last interaction.

    Returns:
        Decayed weight clamped to [0.0, 1.0].
    """
    if elapsed_days < 0:
        raise ValueError(f"elapsed_days must be non-negative, got {elapsed_days}")
    decayed = initial_weight * math.exp(-DECAY_LAMBDA * elapsed_days)
    return max(0.0, min(1.0, decayed))


def elapsed_days_between(older_ts: float, newer_ts: float) -> float:
    """Convert two UNIX timestamps to elapsed days (older → newer)."""
    seconds_per_day = 86_400.0
    return (newer_ts - older_ts) / seconds_per_day
