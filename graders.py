"""
FlowState RL — Hackathon Task Graders
======================================
Per spec/04_reward_shaping.md §3, the final grader score:
  1. Sums cumulative episode rewards.
  2. Normalises by the theoretical max reward.
  3. STRICTLY clamps to (0.001, 0.999) — the hackathon validator
     rejects scores that are exactly 0.0 or 1.0.
  4. Burns out (fatigue >= 1.0) caps the score at 0.001.

These graders are referenced in openenv.yaml under each task and
can also be called directly from inference.py for local validation.
"""

from typing import List


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _compute_score(
    rewards: List[float],
    final_fatigue: float,
    theoretical_max: float,
) -> float:
    """
    Shared scoring pipeline used by all three task graders.

    Args:
        rewards:          List of per-step rewards from the episode.
        final_fatigue:    env.sim_state["fatigue_level"] at episode end.
        theoretical_max:  Maximum achievable cumulative reward for this task.

    Returns:
        A float strictly in (0.001, 0.999).
    """
    if final_fatigue >= 1.0:
        return 0.01

    total = sum(rewards)

    if theoretical_max <= 0:
        return 0.01

    # Normalise
    raw = total / theoretical_max

    # Strict open-interval clamp: 0.0 and 1.0 are both INVALID per validator
    score = max(0.01, min(0.99, round(raw, 2)))
    return score


# ---------------------------------------------------------------------------
# Task-specific graders
# ---------------------------------------------------------------------------

def grade_easy(rewards: List[float], final_fatigue: float) -> float:
    """
    Easy task grader.
    1 goal, 2 hrs planned, 20 max steps.
    Theoretical max ≈ sum of best-case per-step rewards over 20 steps
    with productivity climbing to ~1.0 and fatigue staying low.
    We use 10 steps × 0.95 (typical peak) as a conservative reference.
    """
    theoretical_max = 8.0  # 10 steps × ~0.80 achievable per step
    return _compute_score(rewards, final_fatigue, theoretical_max)


def grade_medium(rewards: List[float], final_fatigue: float) -> float:
    """
    Medium task grader.
    2 goals (4 hrs each), moderate starting fatigue (0.2), 20 max steps.
    Harder: requires balancing two goals + managing extra fatigue.
    """
    theoretical_max = 6.0  # slightly lower because of fatigue headroom
    return _compute_score(rewards, final_fatigue, theoretical_max)


def grade_hard(rewards: List[float], final_fatigue: float) -> float:
    """
    Hard task grader.
    3 goals (6 hrs each), high starting fatigue (0.5), 20 max steps.
    Agent starts already half-burned out — scoring is generous to reflect
    the much harder starting conditions.
    """
    theoretical_max = 4.0  # lower bar because of brutal starting state
    return _compute_score(rewards, final_fatigue, theoretical_max)


# ---------------------------------------------------------------------------
# Registry — used by openenv.yaml task grader resolution
# ---------------------------------------------------------------------------

GRADERS = {
    "easy":   grade_easy,
    "medium": grade_medium,
    "hard":   grade_hard,
}


def grade(task_id: str, rewards: List[float], final_fatigue: float) -> float:
    """
    Dispatch to the correct grader by task_id.

    Args:
        task_id:       One of 'easy', 'medium', 'hard'.
        rewards:       List of per-step rewards collected during the episode.
        final_fatigue: fatigue_level at the end of the episode.

    Returns:
        A float strictly in (0.001, 0.999).

    Raises:
        ValueError: If task_id is not recognised.
    """
    if task_id not in GRADERS:
        raise ValueError(f"Unknown task_id '{task_id}'. Expected one of {list(GRADERS)}")
    return GRADERS[task_id](rewards, final_fatigue)
