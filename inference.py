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
from graders import grade

TASK_NAME = "FlowState Scheduling"
BENCHMARK = "flow_state_rl"

# Default task to run. The grader will test all 3 sequentially, but
# for a single inference run we default to "easy".
DEFAULT_TASK = os.getenv("TASK_ID", "easy")


def log_start(task: str, env_name: str, model: str) -> None:
    payload = json.dumps({"task": task, "env": env_name, "model": model})
    print(f"[START] {payload}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    payload = json.dumps({
        "step": step,
        "action": action,
        "reward": round(reward, 2),
        "done": done,
        "error": error
    })
    print(f"[STEP] {payload}", flush=True)


def log_end(success: bool, steps: int, rewards: List[float], score: float) -> None:
    payload = json.dumps({
        "success": success,
        "steps": steps,
        "score": round(score, 2),
        "rewards": [round(r, 2) for r in rewards]
    })
    print(f"[END] {payload}", flush=True)


def run_episode(env: FlowStateEnv, client: OpenAI, model_name: str, task_id: str) -> tuple:
    """
    Runs a single episode of the given task.

    Returns:
        (rewards: List[float], final_fatigue: float, success: bool, steps: int)
    """
    obs = env.reset(task_id=task_id)
    rewards: List[float] = []
    step_count = 0
    success    = False

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
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=200,
        )
        action_str = response.choices[0].message.content.strip()

        if action_str.startswith("```json"):
            action_str = action_str.replace("```json", "").replace("```", "").strip()
        elif action_str.startswith("```"):
            action_str = action_str.replace("```", "").strip()

        try:
            action_dict = json.loads(action_str)
            action      = BlockAction(**action_dict)
            obs         = env.step(action)

            reward    = float(obs.reward)
            done      = obs.done
            env_error = str(obs.error) if obs.error else None

            if done and not env_error:
                success = True

        except Exception as e:
            reward    = 0.01
            done      = True
            env_error = f"Action parsing failed: {e}"
            success   = False

        rewards.append(reward)
        log_step(step=step, action=action_str, reward=reward, done=done, error=env_error)

        if done:
            break

    final_fatigue = env.sim_state.get("fatigue_level", 0.0)
    return rewards, final_fatigue, success, step_count


def main():
    rewards: List[float] = []
    step_count = 0
    success    = False
    env        = None

    try:
        # ----------------------------------------------------------------
        # 1. API Initialization
        #    Spec §4: must accept API_BASE_URL, MODEL_NAME, HF_TOKEN.
        #    Grader injects API_KEY to track proxy calls — fall back to
        #    HF_TOKEN for local dev so both environments work.
        # ----------------------------------------------------------------
        API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
        MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4.1-mini")
        HF_TOKEN     = os.getenv("HF_TOKEN")
        API_KEY      = os.getenv("API_KEY") or HF_TOKEN  # grader uses API_KEY

        if API_KEY is None:
            raise ValueError(
                "No API key found. Set API_KEY (injected by grader) "
                "or HF_TOKEN (for local dev)."
            )

        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

        # ----------------------------------------------------------------
        # 2. Run episode for the configured tasks
        # ----------------------------------------------------------------
        tasks_to_run = [DEFAULT_TASK] if os.getenv("TASK_ID") else ["easy", "medium", "hard"]

        for tid in tasks_to_run:
            env = FlowStateEnv()
            try:
                log_start(task=tid, env_name=BENCHMARK, model=MODEL_NAME)
                rewards, final_fatigue, success, step_count = run_episode(
                    env, client, MODEL_NAME, tid
                )

                # ----------------------------------------------------------------
                # 3. Compute final grader score (strictly in (0.01, 0.99))
                # ----------------------------------------------------------------
                final_score = grade(tid, rewards, final_fatigue)
            except Exception as e:
                print(f"[DEBUG] Execution error on {tid}: {e}", file=sys.stderr, flush=True)
                final_score = 0.01
                success = False
                step_count = 0
                rewards = []
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception as e:
                        print(f"[DEBUG] Env close error: {e}", file=sys.stderr, flush=True)
                
                log_end(success=success, steps=step_count, rewards=rewards, score=final_score)

    except Exception as general_e:
        print(f"[DEBUG] Global execution error: {general_e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()