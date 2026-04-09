import os
import sys
import json
import asyncio
from openai import OpenAI
from server.env import FlowStateEnv
from models import BlockAction

def log_start(task: str, env_name: str, model: str):
    print("[START] " + json.dumps({"task": task, "env": env_name, "model": model}))

def log_step(step: int, action: str, reward: float, done: bool, error):
    print("[STEP] " + json.dumps({
        "step": step, 
        "action": action, 
        "reward": round(float(reward), 2), 
        "done": done, 
        "error": error
    }))

def log_end(success: bool, steps: int, score: float, rewards: list):
    formatted_rewards = [round(float(r), 2) for r in rewards]
    print("[END] " + json.dumps({
        "success": success, 
        "steps": steps, 
        "score": round(float(max(0.0, min(1.0, score))), 2), 
        "rewards": formatted_rewards
    }))

async def main():
    api_base = os.getenv("API_BASE_URL", "https://router.huggingface.co/hf-inference/v1/")
    model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
    
    api_key = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
    if not api_key:
        raise ValueError("HF_TOKEN environment variable is required")

    client = OpenAI(base_url=api_base, api_key=api_key)
    
    log_start("FlowState Agentic Evaluation", "flow_state_rl", model_name)
    
    env = FlowStateEnv()
    obs = env.reset()
    
    rewards = []
    done = False
    step_count = 0
    error_msg = None
    
    try:
        for i in range(1, 11):
            step_count = i
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a FlowState agent. Output ONLY valid JSON matching the BlockAction schema. Do not include markdown."
                    },
                    {
                        "role": "user", 
                        "content": f"Current Observation: {json.dumps(obs.model_dump())}\nChoose the best next action."
                    }
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            action_str = response.choices[0].message.content.strip()
            
            # Robust markdown cleaning
            if action_str.startswith("```json"):
                action_str = action_str.replace("```json", "", 1)
            elif action_str.startswith("```"):
                action_str = action_str.replace("```", "", 1)
            if action_str.endswith("```"):
                action_str = action_str[:-3]
            action_str = action_str.strip()
                
            action_dict = json.loads(action_str)
            action = BlockAction(**action_dict)
            
            obs = env.step(action)
            reward = obs.reward
            done = obs.done
            error_msg = obs.error
            
            rewards.append(reward)
            log_step(step=step_count, action=action_str, reward=reward, done=done, error=error_msg)
            
            if done:
                break

    except Exception as e:
        print(f"\n[DEBUG ERROR] Episode crashed at step {step_count}: {str(e)}\n", file=sys.stderr)
        error_msg = str(e)
    finally:
        avg_score = sum(rewards) / len(rewards) if rewards else 0.0
        is_success = (not bool(error_msg)) and len(rewards) > 0
        log_end(success=is_success, steps=step_count, score=avg_score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())