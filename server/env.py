from openenv_core import Environment, State
from models import EnergyState, GoalProgress, BlockAction, BlockObservation

class FlowStateEnv(Environment):
    def __init__(self):
        super().__init__()
        self.step_count = 0
        self.sim_state = self._get_initial_state()

    @property
    def state(self) -> State:
        return State(
            episode_id="flow_state_v1",
            step_count=self.step_count
        )

    def _get_initial_state(self) -> dict:
        """
        Returns the initial pristine state for the environment.
        Follows the 24-hour time conservation rule initially set up.
        """
        return {
            "day": 1,
            "fatigue_level": 0.0,
            "focus_score": 1.0,
            "constant_block": 10.0,
            "external_block": 4.0,
            "goal_block": 7.5,
            "break_block": 1.0,
            "reserve_block": 1.0,  # 10 + 4 + 7.5 + 1 + 1 = 23.5 (free time omitted per schema)
            "energy": {
                "high_used": 0.0,
                "medium_used": 0.0,
                "low_used": 0.0,
            },
            "goals": {}
        }

    def _build_observation(self, reward: float, done: bool, error: str = None) -> BlockObservation:
        """
        Maps the current internal state into the strictly typed Pydantic BlockObservation model.
        """
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

    def _compute_reward(self) -> float:
        """
        Calculates the productivity-based reward, penalized by fatigue.
        """
        planned_total = 0.0
        completed_total = 0.0
        
        for goal in self.sim_state.get("goals", {}).values():
            if isinstance(goal, dict):
                planned_total += float(goal.get("planned", 0.0))
                completed_total += float(goal.get("completed", 0.0))
            else:
                planned_total += float(getattr(goal, "planned", 0.0))
                completed_total += float(getattr(goal, "completed", 0.0))
        
        productivity = (completed_total / planned_total) if planned_total > 0 else 0.0
        fatigue_penalty = self.sim_state.get("fatigue_level", 0.0) * 0.5
        
        return round(productivity - fatigue_penalty, 4)

    def reset(self, task_id: str = None, **kwargs) -> BlockObservation:
        """
        Wipes the state completely and initializes based on task difficulty.
        Supports: 'easy', 'medium', 'hard'. Defaults to 'easy'.
        """
        self.step_count = 0
        self.sim_state = self._get_initial_state()

        # Default to easy if no task_id provided
        if task_id is None:
            task_id = "easy"

        if task_id == "easy":
            # 1 goal, 2 hours planned, full energy (focus_score=1.0)
            self.sim_state["goals"] = {
                "Goal_1": {"planned": 2.0, "completed": 0.0}
            }
            self.sim_state["focus_score"] = 1.0
            self.sim_state["fatigue_level"] = 0.0

        elif task_id == "medium":
            # 2 goals, 3 hours each, moderate starting fatigue (focus_score=0.8)
            self.sim_state["goals"] = {
                "Goal_1": {"planned": 3.0, "completed": 0.0},
                "Goal_2": {"planned": 3.0, "completed": 0.0}
            }
            self.sim_state["focus_score"] = 0.8
            self.sim_state["fatigue_level"] = 0.2

        elif task_id == "hard":
            # 3 goals, 5 hours each, low starting energy (focus_score=0.5)
            self.sim_state["goals"] = {
                "Goal_1": {"planned": 5.0, "completed": 0.0},
                "Goal_2": {"planned": 5.0, "completed": 0.0},
                "Goal_3": {"planned": 5.0, "completed": 0.0}
            }
            self.sim_state["focus_score"] = 0.5
            self.sim_state["fatigue_level"] = 0.5

        return self._build_observation(reward=0.0, done=False)

    def step(self, action: BlockAction) -> BlockObservation:
        """
        Takes an action, updates the state, and calculates reward.
        """
        self.step_count += 1
        
        # 1. Idempotency Check
        if self.sim_state.get("fatigue_level", 0.0) >= 1.0:
            return self._build_observation(reward=0.0, done=True, error="User is burned out.")
            
        try:
            # 3. Action Processing
            adjust_blocks = getattr(action, "adjust_blocks", {})
            break_adj = adjust_blocks.get("break_block", 0.0) if hasattr(adjust_blocks, "get") else 0.0
            
            # Ensure break_block doesn't drop below 0
            current_break = self.sim_state.get("break_block", 1.0)
            self.sim_state["break_block"] = max(0.0, current_break + float(break_adj))
            
            # 4. Progress Goals
            adjust_goal = getattr(action, "adjust_goal", {})
            if hasattr(adjust_goal, "items"):
                for goal_name, adjustment in adjust_goal.items():
                    goals_dict = self.sim_state.get("goals", {})
                    if goal_name in goals_dict:
                        goal = goals_dict[goal_name]
                        if isinstance(goal, dict):
                            current_completed = goal.get("completed", 0.0)
                            goal["completed"] = current_completed + (1.0 + float(adjustment))
                        else:
                            current_completed = getattr(goal, "completed", 0.0)
                            setattr(goal, "completed", current_completed + (1.0 + float(adjustment)))
                            
            # 5. Fatigue Math
            fatigue_delta = (0.1 * 2.0) - (0.08 * self.sim_state["break_block"])
            current_fatigue = self.sim_state.get("fatigue_level", 0.0)
            new_fatigue = max(0.0, min(1.0, current_fatigue + fatigue_delta))
            self.sim_state["fatigue_level"] = new_fatigue
            
            # 6. Reward Math
            reward = self._compute_reward()
            
            # 7. Termination
            done = self.sim_state["fatigue_level"] >= 1.0
            
            # 8. Return
            return self._build_observation(reward=reward, done=done)
            
        except Exception as e:
            # 2. Graceful Degradation
            return self._build_observation(reward=-1.0, done=False, error=str(e))