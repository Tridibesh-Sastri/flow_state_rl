import os
import json
from typing import List, Optional
from openai import OpenAI

# Direct instantiation of your environment for the baseline script
from server.env import FlowStateEnv
from models import BlockAction

# --- CONFIGURATION ---
# Strictly use injected variables first, fallback to HF Router for local testing
API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
API_BASE_URL = os.environ.get("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.environ.get("MODEL_NAME") or "meta-llama/Llama-3.1-8B-Instruct"

TASK_NAME = "FlowState Scheduling"
BENCHMARK = "flow_state_rl"
MAX_STEPS = 5
SUCCESS_SCORE_THRESHOLD = 0.5  # Adjust based on your reward logic

# --- STRICT LOGGING FORMATTERS ---
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    
    # CRITICAL: Strip newlines from action so it doesn't break the single-line stdout rule
    action_clean = action.replace("\n", " ").replace("\r", " ")
    
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# --- MAIN EXECUTION ---
def main():
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        # 1. Initialize strictly via the required variables
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        
        # 2. Instantiate Environment
        env = FlowStateEnv()
        obs = env.reset()
        
        for step in range(1, MAX_STEPS + 1):
            steps_taken = step
            
            # 3. Formulate Prompt
            system_prompt = "You are a FlowState scheduling agent. Output ONLY valid JSON matching the BlockAction schema. Do not include markdown formatting."
            user_prompt = f"Current Observation: {json.dumps(obs.model_dump())}\nChoose the best next action."
            
            action_str = "{}"
            
            # 4. LLM Call via Proxy
            try:
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
            except Exception as e:
                print(f"[DEBUG] Model request failed: {e}", flush=True)
                # Let it fall through so the environment can flag the error and close gracefully

            # 5. Clean Markdown (if the model ignores instructions)
            if action_str.startswith("```json"):
                action_str = action_str.replace("```json", "").replace("```", "").strip()
            elif action_str.startswith("```"):
                action_str = action_str.replace("```", "").strip()
            
            # 6. Parse and Step
            try:
                action_dict = json.loads(action_str)
                action = BlockAction(**action_dict)
                obs = env.step(action)
                
                reward = float(obs.reward)
                done = obs.done
                env_error = str(obs.error) if obs.error else None
                
            except Exception as e:
                # If JSON parsing fails, penalize and end episode
                reward = 0.0
                done = True
                env_error = f"Action parsing failed: {e}"
                
            rewards.append(reward)
            log_step(step=step, action=action_str, reward=reward, done=done, error=env_error)
            
            if done:
                break
                
        # 7. Final Scoring
        avg_score = sum(rewards) / len(rewards) if rewards else 0.0
        score = max(0.0, min(1.0, avg_score))  # Clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode crashed unexpectedly: {e}", flush=True)
    finally:
        # 8. Ensure END is ALWAYS called
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()