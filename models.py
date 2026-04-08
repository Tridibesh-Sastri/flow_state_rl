import json
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator

class EnergyState(BaseModel):
    high_used: float = Field(default=0.0, description="High energy hours used")
    medium_used: float = Field(default=0.0, description="Medium energy hours used")
    low_used: float = Field(default=0.0, description="Low energy hours used")

class GoalProgress(BaseModel):
    planned: float = Field(default=0.0, description="Planned hours for the goal")
    completed: float = Field(default=0.0, description="Completed hours for the goal")

class BlockAction(BaseModel):
    adjust_goal: Dict[str, Any] = Field(default_factory=dict, description="Goal adjustments")
    adjust_blocks: Dict[str, Any] = Field(default_factory=dict, description="Block adjustments")
    energy_shift: Dict[str, Any] = Field(default_factory=dict, description="Energy shift allocations")

    @model_validator(mode='before')
    @classmethod
    def parse_strings_to_dicts(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for field in ['adjust_goal', 'adjust_blocks', 'energy_shift']:
                if field in data and isinstance(data[field], str):
                    try:
                        # Parse string to dict, default to {} if empty
                        data[field] = json.loads(data[field]) if data[field].strip() else {}
                    except Exception:
                        data[field] = {}
        return data

class BlockObservation(BaseModel):
    # Standard OpenEnv fields
    reward: float = Field(default=0.0, description="The calculated reward for the current step")
    done: bool = Field(default=False, description="Whether the episode has reached a terminal state")
    error: Optional[str] = Field(default=None, description="Error message for feedback")
    
    # Custom state fields
    day: int = Field(default=1, description="Current day")
    fatigue_level: float = Field(default=0.0, ge=0.0, le=1.0, description="Fatigue level, bounded between 0 and 1")
    focus_score: float = Field(default=1.0, description="Current focus score")
    constant_block: float = Field(default=10.0, description="Time spent on constant tasks")
    external_block: float = Field(default=0.0, description="Time spent on external events")
    goal_block: float = Field(default=0.0, description="Time spent on productive work")
    break_block: float = Field(default=0.0, description="Time spent on breaks")
    reserve_block: float = Field(default=0.0, description="Time reserved as buffer")
    
    # Nested fields
    energy: EnergyState = Field(default_factory=EnergyState, description="Energy consumption state")
    goals: Dict[str, GoalProgress] = Field(default_factory=dict, description="Progress states of goals")

# Aliases for compatibility with other files if needed
FlowStateRlAction = BlockAction
FlowStateRlObservation = BlockObservation
