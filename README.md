---
title: FlowState RL
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
---


# FlowState RL: Agentic Productivity Optimizer 🧠🚀

FlowState is a high-fidelity Reinforcement Learning environment designed to train agents in the art of human-centric schedule optimization. Unlike simple toy environments, FlowState simulates the complex interplay between **Cognitive Focus**, **Fatigue Accumulation**, and **Task Completion**.

## 🌟 Key Features
* **Dynamic Fatigue Modeling**: Uses a physics-inspired decay function where high-intensity work increases fatigue, and breaks provide non-linear recovery.
* **Reward Shaping**: The environment rewards productivity but applies heavy penalties for burnout (over-fatigue), forcing the AI to learn the value of rest.
* **Multi-Task Management**: Supports parallel goal tracking with adjustable priorities.
* **OpenEnv Compliant**: Fully compatible with the Meta PyTorch OpenEnv standard for automated agent evaluation.

## 📊 Environment Details
### Action Space
The agent provides a `BlockAction` dictionary:
* `adjust_goal`: Assign work hours to specific objectives.
* `adjust_blocks`: Reallocate time between work, breaks, and reserve blocks.
* `energy_shift`: Manage high/medium/low energy usage across the day.

### Observation Space
The agent receives a `BlockObservation`:
* `fatigue_level`: Current exhaustion metric (0.0 to 1.0).
* `focus_score`: Efficiency multiplier based on current energy.
* `goals`: Real-time progress tracking for all active objectives.

## 🚀 Quick Start
```python
from flow_state_rl import FlowStateRlAction, FlowStateRlEnv

async with FlowStateRlEnv(base_url="[https://trixion-flow-state-rl.hf.space](https://trixion-flow-state-rl.hf.space)") as env:
    result = await env.reset()
    # Execute a strategic work/break step
    result = await env.step(FlowStateRlAction(
        adjust_goal={"Goal_1": 0.5},
        adjust_blocks={"break_block": 0.5}
    ))
    print(f"Reward: {result.reward} | Fatigue: {result.observation.fatigue_level}")