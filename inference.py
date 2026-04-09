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
    api_base = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
    model_name = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
    
    # Strictly follow the validator's explicit instructions
    api_key = os.environ.get("API_KEY")
    if not api_key:
        api_key = os.environ.get("HF_TOKEN")
        
    client = OpenAI(base_url=api_base, api_key=api_key)
    
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
        
        # REAL LLM CALL - NO SAFETY NET
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a FlowState scheduling agent. Output ONLY valid JSON matching the BlockAction schema. Do not include markdown formatting."
                },
                {
                    "role": "user", 
                    "content": f"Current Observation: {json.dumps(obs.model_dump())}\nChoose the best next action."
                }
            ]
        )
        
        action_str = response.choices[0].message.content.strip()
        
        # Clean markdown if the model includes it
        if action_str.startswith("```json"):
            action_str = action_str.replace("```json", "").replace("```", "").strip()
        elif action_str.startswith("```"):
            action_str = action_str.replace("```", "").strip()
            
        action_dict = json.loads(action_str)
        action = BlockAction(**action_dict)
        
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