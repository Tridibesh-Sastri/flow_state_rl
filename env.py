import math
from openenv_core import Environment, State
from models import BlockAction, BlockObservation


class FlowStateEnv(Environment):
    """
    FlowState RL Environment — Agentic Productivity Optimizer.

    Simulates a human's 24-hour workday using a multi-objective reward
    function (per spec/04_reward_shaping.md) that tracks:
      - Productivity (P)
      - Energy Alignment (A)  — circadian sine-wave model
      - Context / R.E.A.L. Bonus (C)
      - Overload Penalty (O)
      - Fatigue Penalty (F)

    Rewards are dense and strictly in (0.001, 0.999) so they are always
    accepted by the hackathon validator (which rejects 0.0 and 1.0).
    """

    # Multi-objective reward weights (from spec/04_reward_shaping.md)
    W_P = 0.4   # Productivity
    W_A = 0.3   # Energy Alignment
    W_C = 0.1   # Context (R.E.A.L. Rewire Bonus)
    W_O = 0.5   # Overload Penalty
    W_F = 0.5   # Fatigue Penalty

    def __init__(self):
        super().__init__()
        self.step_count = 0
        self.sim_state = self._get_initial_state()
        self._prev_break_block = 1.0   # tracks previous step's break for R.E.A.L. bonus

    @property
    def state(self) -> State:
        return State(
            episode_id="flow_state_v1",
            step_count=self.step_count
        )

    def _get_initial_state(self) -> dict:
        """
        Returns the initial pristine state for the environment.
        Follows the 24-hour time conservation rule.
        """
        return {
            "day": 1,
            "fatigue_level": 0.0,
            "focus_score": 1.0,
            # Time blocks (hours)
            "constant_block": 10.0,    # sleep + meals (fixed)
            "external_block": 4.0,     # meetings/classes (fixed)
            "goal_block": 7.5,         # productive work
            "break_block": 1.0,        # recovery
            "reserve_block": 1.0,      # emergency buffer
            # Energy usage ledger
            "energy": {
                "high_used": 0.0,
                "medium_used": 0.0,
                "low_used": 0.0,
            },
            "goals": {}
        }

    def _build_observation(self, reward: float, done: bool, error: str = None) -> BlockObservation:
        """Maps the current internal state into the typed Pydantic model."""
        return BlockObservation(
            reward=reward,
            done=done,
            error=error,
            day=self.sim_state.get("day", 1),
            fatigue_level=self.sim_state.get("fatigue_level", 0.0),
            focus_score=self.sim_state.get("focus_score", 1.0),
            constant_block=self.sim_state.get("constant_block", 10.0),
            external_block=self.sim_state.get("external_block", 4.0),
            goal_block=self.sim_state.get("goal_block", 7.5),
            break_block=self.sim_state.get("break_block", 1.0),
            reserve_block=self.sim_state.get("reserve_block", 1.0),
            energy=self.sim_state.get("energy", {"high_used": 0.0, "medium_used": 0.0, "low_used": 0.0}),
            goals=self.sim_state.get("goals", {})
        )

    # ------------------------------------------------------------------
    # Reward helper: Circadian energy level at a given hour (0-24)
    # E(t) = 0.5 + 0.4 * sin(2π * (t - 8.0) / 24.0)
    # ------------------------------------------------------------------
    @staticmethod
    def _circadian_energy(hour: float = 14.0) -> float:
        """Returns the circadian energy level [0.1 – 0.9] for a given hour."""
        return 0.5 + 0.4 * math.sin(2 * math.pi * (hour - 8.0) / 24.0)

    def _compute_reward(
        self,
        goal_hours_scheduled: float,
        break_hours_delta: float,
        had_context_switch: bool,
        overload: bool,
    ) -> float:
        """
        Multi-objective reward formula (spec/04_reward_shaping.md §1-2).

        R = (W_P * P) + (W_A * A) + (W_C * C) - (W_O * O) - (W_F * F)

        Then clamped strictly to (0.001, 0.999) so the hackathon validator
        never sees exactly 0.0 or 1.0.
        """
        # P — Productivity: fraction of total planned hours completed this step
        planned_total = 0.0
        completed_total = 0.0
        for goal in self.sim_state.get("goals", {}).values():
            if isinstance(goal, dict):
                planned_total += float(goal.get("planned", 0.0))
                completed_total += float(goal.get("completed", 0.0))
            else:
                planned_total += float(getattr(goal, "planned", 0.0))
                completed_total += float(getattr(goal, "completed", 0.0))

        P = min(1.0, completed_total / planned_total) if planned_total > 0 else 0.0

        # A — Energy Alignment: circadian penalty/bonus
        # Use step 14:00 as proxy for peak; in a real sim you'd track time-of-day
        energy_now = self._circadian_energy(14.0)   # peak alignment for work
        if goal_hours_scheduled > 0 and energy_now > 0.7:
            A = 1.0   # scheduling work during peak → bonus
        elif goal_hours_scheduled > 0 and energy_now < 0.4:
            A = -1.0  # scheduling work during dip → penalty
        elif break_hours_delta > 0 and energy_now < 0.4:
            A = 0.8   # scheduling break during energy dip → smart
        else:
            A = 0.0

        # C — Context / R.E.A.L. Rewire Bonus
        C = 1.0 if had_context_switch else 0.0

        # O — Overload penalty
        O = 1.0 if overload else 0.0

        # F — Fatigue penalty (continuous as fatigue rises)
        F = self.sim_state.get("fatigue_level", 0.0)

        raw = (self.W_P * P) + (self.W_A * A) + (self.W_C * C) \
              - (self.W_O * O) - (self.W_F * F)

        # Normalise from theoretical range [-0.8, 0.8] → [0, 1]
        # raw_min ≈ -0.8 (full overload + burnout), raw_max ≈ 0.8 (perfect)
        normalised = (raw + 0.8) / 1.6

        # CRITICAL: hackathon validator rejects 0.0 and 1.0 — keep strictly inside
        return round(max(0.001, min(0.999, normalised)), 4)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, task_id: str = None, **kwargs) -> BlockObservation:
        """
        Wipes the state completely and initialises based on task difficulty.
        Supports: 'easy', 'medium', 'hard'. Defaults to 'easy'.
        """
        self.step_count = 0
        self.sim_state = self._get_initial_state()
        self._prev_break_block = 1.0

        if task_id is None:
            task_id = "easy"

        if task_id == "easy":
            # 1 goal, 2 hours planned, full energy, no fatigue
            self.sim_state["goals"] = {
                "Goal_Alpha": {"planned": 2.0, "completed": 0.0}
            }
            self.sim_state["focus_score"] = 1.0
            self.sim_state["fatigue_level"] = 0.0

        elif task_id == "medium":
            # 2 goals, 4 hours each, moderate starting fatigue
            self.sim_state["goals"] = {
                "Goal_Alpha": {"planned": 4.0, "completed": 0.0},
                "Goal_Beta":  {"planned": 4.0, "completed": 0.0},
            }
            self.sim_state["focus_score"] = 0.8
            self.sim_state["fatigue_level"] = 0.2

        elif task_id == "hard":
            # 3 goals, 6 hours each, high starting fatigue
            self.sim_state["goals"] = {
                "Goal_Alpha": {"planned": 6.0, "completed": 0.0},
                "Goal_Beta":  {"planned": 6.0, "completed": 0.0},
                "Goal_Gamma": {"planned": 6.0, "completed": 0.0},
            }
            self.sim_state["focus_score"] = 0.5
            self.sim_state["fatigue_level"] = 0.5

        return self._build_observation(reward=0.001, done=False)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: BlockAction) -> BlockObservation:
        """
        Applies an action, updates the simulation state, computes the
        multi-objective reward, and returns the next observation.
        """
        self.step_count += 1

        # 1. Idempotency: If already burned out, freeze state
        if self.sim_state.get("fatigue_level", 0.0) >= 1.0:
            return self._build_observation(reward=0.001, done=True, error="User is burned out.")

        try:
            adjust_blocks = getattr(action, "adjust_blocks", {}) or {}
            adjust_goal   = getattr(action, "adjust_goal",   {}) or {}

            # ----------------------------------------------------------
            # 2. Break block adjustment
            # ----------------------------------------------------------
            break_adj = float(adjust_blocks.get("break_block", 0.0)) if hasattr(adjust_blocks, "get") else 0.0
            self._prev_break_block = self.sim_state.get("break_block", 1.0)
            new_break = max(0.0, self._prev_break_block + break_adj)
            self.sim_state["break_block"] = new_break

            # ----------------------------------------------------------
            # 3. Progress goals
            # ----------------------------------------------------------
            goal_hours_scheduled = 0.0
            if hasattr(adjust_goal, "items"):
                for goal_name, adjustment in adjust_goal.items():
                    goals_dict = self.sim_state.get("goals", {})
                    if goal_name in goals_dict:
                        goal = goals_dict[goal_name]
                        hours = max(0.0, float(adjustment))
                        goal_hours_scheduled += hours
                        if isinstance(goal, dict):
                            goal["completed"] = max(0.0, goal.get("completed", 0.0) + hours)
                        else:
                            setattr(goal, "completed",
                                    max(0.0, getattr(goal, "completed", 0.0) + hours))

            # ----------------------------------------------------------
            # 4. Fatigue update (spec/03_project_spec.md §4)
            # Fatigue_new = clamp(Fatigue_old + 0.10*High_hrs + 0.05*Med_hrs - 0.08*Break_hrs)
            # We treat all goal work as "high energy" for simplicity.
            # ----------------------------------------------------------
            high_hrs = goal_hours_scheduled if goal_hours_scheduled > 0 else 2.0  # default workload
            break_hrs = self.sim_state["break_block"]
            fatigue_delta = (0.10 * high_hrs) - (0.08 * break_hrs)
            current_fatigue = self.sim_state.get("fatigue_level", 0.0)
            self.sim_state["fatigue_level"] = max(0.0, min(1.0, current_fatigue + fatigue_delta))

            # ----------------------------------------------------------
            # 5. Detect R.E.A.L. context switch bonus:
            #    Agent increased break_block before scheduling goal work
            # ----------------------------------------------------------
            had_context_switch = (break_adj > 0.1 and goal_hours_scheduled > 0)

            # ----------------------------------------------------------
            # 6. Detect overload (sum of hours > 24)
            # ----------------------------------------------------------
            total_hours = (
                self.sim_state.get("constant_block", 10.0)
                + self.sim_state.get("external_block",  4.0)
                + self.sim_state.get("goal_block",      7.5)
                + self.sim_state["break_block"]
                + self.sim_state.get("reserve_block",   1.0)
            )
            overload = total_hours > 24.0

            # ----------------------------------------------------------
            # 7. Compute multi-objective reward
            # ----------------------------------------------------------
            reward = self._compute_reward(
                goal_hours_scheduled=goal_hours_scheduled,
                break_hours_delta=break_adj,
                had_context_switch=had_context_switch,
                overload=overload,
            )

            # ----------------------------------------------------------
            # 8. Termination check
            # ----------------------------------------------------------
            done = self.sim_state["fatigue_level"] >= 1.0

            return self._build_observation(reward=reward, done=done)

        except Exception as e:
            # Graceful degradation — never crash (spec rule §4)
            return self._build_observation(reward=0.001, done=False, error=str(e))