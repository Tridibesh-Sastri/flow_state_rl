import os
import json
import sys
from typing import List, Optional
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Direct instantiation of environment for the baseline script
from env import FlowStateEnv
from models import BlockAction

TASK_NAME = "FlowState Scheduling"
BENCHMARK = "flow_state_rl"

def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={task} env={env_name} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = str(error) if error is not None else "null"
    done_val = "true" if done else "false"
    
    # Strip newlines from action to avoid breaking the single-line requirement
    action_clean = action.replace("\n", " ").replace("\r", " ")
    
    print(f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    success_val = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    
    # CRITICAL FIX: Removed 'score=' to strictly match the regex validator
    print(f"[END] success={success_val} steps={steps} rewards={rewards_str}", flush=True)

def main():
    rewards = []
    step_count = 0
    success = False
    env = None
    
    try:
        # 1. API Initialization (Strict fallback pattern)
        API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
        MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
        HF_TOKEN = os.getenv("HF_TOKEN")
        
        if HF_TOKEN is None:
            raise ValueError("HF_TOKEN environment variable is required")
            
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN
        )
        
        log_start(task=TASK_NAME, env_name=BENCHMARK, model=MODEL_NAME)
        
        env = FlowStateEnv()
        obs = env.reset()
        
        for step in range(1, 21):
            step_count = step
            
            goal_names = list(obs.model_dump().get("goals", {}).keys())
            system_prompt = (
                "You are a FlowState scheduling agent. Your job is to maximize task completion while managing fatigue.\n\n"
                "OUTPUT ONLY valid JSON with EXACTLY these keys:\n"
                "  adjust_goal   - dict mapping goal names to hours to add (e.g. {\"Goal_Alpha\": 0.5})\n"
                "  adjust_blocks - dict with optional key 'break_block' for break adjustments (e.g. {\"break_block\": 0.1})\n"
                "  energy_shift  - dict for energy tier shifts (can be empty {})\n\n"
                "EXAMPLE of a valid action (do work AND take a small break to avoid burnout):\n"
                "{\"adjust_goal\": {\"Goal_Alpha\": 0.5}, \"adjust_blocks\": {\"break_block\": 0.05}, \"energy_shift\": {}}\n\n"
                "RULES:\n"
                "- Only use goal names that appear in the 'goals' field of the observation.\n"
                "- If fatigue_level > 0.7, prioritize increasing break_block instead of working.\n"
                "- Do NOT use keys like 'action', 'task', 'duration', or 'work'. They will be IGNORED.\n"
                "- Do NOT include markdown code fences."
            )
            user_prompt = (
                f"Current Observation: {json.dumps(obs.model_dump())}\n"
                f"Available goals: {goal_names}\n"
                f"Choose the best next action as a single JSON object."
            )
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            action_str = response.choices[0].message.content.strip()
            
            if action_str.startswith("```json"):
                action_str = action_str.replace("```json", "").replace("```", "").strip()
            elif action_str.startswith("```"):
                action_str = action_str.replace("```", "").strip()
            
            try:
                action_dict = json.loads(action_str)
                action = BlockAction(**action_dict)
                obs = env.step(action)
                
                reward = float(obs.reward)
                done = obs.done
                env_error = str(obs.error) if obs.error else None
                
                # Explicit Success Tracking
                if done and not env_error:
                    success = True
                    
            except Exception as e:
                reward = 0.0
                done = True
                env_error = f"Action parsing failed: {e}"
                success = False
                
            rewards.append(reward)
            log_step(step=step, action=action_str, reward=reward, done=done, error=env_error)
            
            if done:
                break
                
    except Exception as e:
        # CRITICAL FIX: Direct debug errors to stderr so they don't break the stdout regex scraper
        print(f"[DEBUG] Execution error: {e}", file=sys.stderr, flush=True)
    finally:
        # CRITICAL FIX: Ensure environment is closed before emitting [END]
        if env is not None:
            try:
                env.close()
            except Exception as e:
                print(f"[DEBUG] Env close error: {e}", file=sys.stderr, flush=True)
                
        # Emit final telemetry
        log_end(success=success, steps=step_count, rewards=rewards)

if __name__ == "__main__":
    main()