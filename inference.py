import os
import json
import asyncio
from openai import OpenAI

# Direct instantiation of environment for the baseline script
from server.env import FlowStateEnv
from models import BlockAction

def log_start(task: str, env_name: str, model: str):
    """Logs the START event exactly to specifications."""
    print("[START] " + json.dumps({
        "task": task,
        "env": env_name,
        "model": model
    }))

def log_step(step: int, action: str, reward: float, done: bool, error):
    """Logs the STEP event exactly to specifications."""
    print("[STEP] " + json.dumps({
        "step": step,
        "action": action,
        "reward": float(reward),
        "done": bool(done),
        "error": error
    }))

def log_end(success: bool, steps: int, score: float, rewards: list):
    """Logs the END event exactly to specifications."""
    print("[END] " + json.dumps({
        "success": bool(success),
        "steps": int(steps),
        "score": float(max(0.0, min(1.0, score))),
        "rewards": rewards
    }))

async def main():
    # Ingest environment variables
    api_base = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("MODEL_NAME", "baseline-model")
    hf_token = os.environ.get("HF_TOKEN", "dummy-token")
    
    # Instantiate OpenAI client
    client = OpenAI(base_url=api_base, api_key=hf_token)
    
    # Log start
    log_start("FlowState Baseline Verification", "flow_state_rl", model_name)
    
    # Instantiate environment locally
    env = FlowStateEnv()
    obs = env.reset()
    
    rewards = []
    done = False
    step_count = 0
    
    # Loop for a maximum of 5 steps
    for i in range(1, 6):
        step_count = i
        
        # MOCK LLM CALL
        # In a real run, you'd send `obs.model_dump_json()` inside the prompt.
        action_dict = {
            "adjust_goal": {f"Mock Goal {i}": 0.5},
            "adjust_blocks": {"break_block": 1.0},
            "energy_shift": {}
        }
        
        try:
            action = BlockAction(**action_dict)
            action_str = json.dumps(action_dict)
        except Exception:
            action = BlockAction()
            action_str = "{}"
        
        # Step through Environment
        obs = env.step(action)
        reward = obs.reward
        done = obs.done
        error = obs.error
        
        rewards.append(reward)
        
        # Log Step
        log_step(step=step_count, action=action_str, reward=reward, done=done, error=error)
        
        if done:
            break

    # Calculate clamped score
    avg_score = sum(rewards) / len(rewards) if rewards else 0.0
    clamped_score = max(0.0, min(1.0, avg_score))

    # Log End
    log_end(success=(not bool(obs.error)), steps=step_count, score=clamped_score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())