import os
import json
from typing import List, Optional
from openai import OpenAI

from server.env import FlowStateEnv
from models import BlockAction

TASK_NAME = "FlowState Scheduling"
BENCHMARK = "flow_state_rl"

def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={task} env={env_name} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = str(error) if error else "null"
    done_val = str(done).lower()
    
    # Strip newlines from action to avoid breaking the single-line requirement
    action_clean = action.replace("\n", " ").replace("\r", " ")
    
    print(f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    success_val = str(success).lower()
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    
    print(f"[END] success={success_val} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def main():
    rewards = []
    step_count = 0
    obs = None
    
    try:
        # The validator explicitly checks for these exact keys
        client = OpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"]
        )
        model_name = os.environ["MODEL_NAME"]
        
        log_start(task=TASK_NAME, env_name=BENCHMARK, model=model_name)
        
        env = FlowStateEnv()
        obs = env.reset()
        
        for step in range(1, 6):
            step_count = step
            
            system_prompt = "You are a FlowState scheduling agent. Output ONLY valid JSON matching the BlockAction schema. Do not include markdown formatting."
            user_prompt = f"Current Observation: {json.dumps(obs.model_dump())}\nChoose the best next action."
            
            action_str = "{}"
            
            response = client.chat.completions.create(
                model=model_name,
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
                
            except Exception as e:
                reward = 0.0
                done = True
                env_error = f"Action parsing failed: {e}"
                # If an error happens, we attach it to the obs so finally picks it up correctly
                if hasattr(obs, 'error'):
                    obs.error = env_error
                else:
                    # Very edge case
                    class MockObs:
                        error = env_error
                    obs = MockObs()
                
            rewards.append(reward)
            log_step(step=step, action=action_str, reward=reward, done=done, error=env_error)
            
            if done:
                break
                
    except Exception as e:
        print(f"[DEBUG] Execution error: {e}", flush=True)
    finally:
        # This block ensures the evaluator ALWAYS receives the final state
        avg_score = sum(rewards) / len(rewards) if rewards else 0.0
        clamped_score = max(0.0, min(1.0, avg_score))
        
        is_success = False
        if obs and getattr(obs, 'error', True) is None:
            is_success = True
            
        log_end(success=is_success, steps=step_count, score=clamped_score, rewards=rewards)

if __name__ == "__main__":
    main()