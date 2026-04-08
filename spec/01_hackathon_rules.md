# SYSTEM INSTRUCTION: HACKATHON RULES & CONSTRAINTS

**To the AI Agent reading this:** You are acting as the Lead Systems Engineer for the "Meta PyTorch OpenEnv Hackathon." If you violate any of the rules in this document, the project will be automatically disqualified by the automated grading scripts. Treat these rules as immutable laws of physics.

---

## 1. THE PRIME DIRECTIVES (NON-NEGOTIABLES)
* **Real-World Application:** The environment MUST simulate a real-world task. Do NOT build a toy game (e.g., no Sudoku, no Wordle, no Gridworlds).
* **Framework:** You MUST use the OpenEnv interface standard (`step()`, `reset()`, `state()`).
* **Hardware Limits:** The environment and inference script MUST be capable of running on a machine with exactly **2 vCPUs and 8GB RAM**.
* **Time Limits:** The total runtime of the inference script MUST be **under 20 minutes**. Code must be highly optimized.

---

## 2. ARCHITECTURE & DATA STRUCTURES
* **Pydantic Serialization:** All `Action` and `Observation` schemas in `models.py` MUST be built using `pydantic.BaseModel`. Do not use standard Python dataclasses or raw dictionaries for the I/O layer.
* **Manifest:** The root directory MUST contain an `openenv.yaml` file properly linking to the environment class.
* **Environment Base Class:** The core logic in `server/env.py` MUST inherit from the OpenEnv `Environment` base class.

---

## 3. TASKS & REWARD SHAPING REQUIREMENTS
You must program the environment to support a minimum of **3 Difficulty Tasks** (Easy, Medium, Hard).

* **Graders:** Each task MUST have an automated grader that evaluates the final state and returns a normalized score strictly between `0.0` and `1.0`. 
* **Partial Progress:** The `step()` function MUST return meaningful intermediate rewards. 
    * *Agent Instruction:* Do NOT use sparse rewards (e.g., giving 0.0 for 99 steps and 1.0 at the end). Use dense rewards to guide the RL agent (e.g., +0.1 for scheduling a task without overlap, -0.5 for causing fatigue).

---

## 4. INFERENCE SCRIPT STRICT PROTOCOL (`inference.py`)
This is the most strictly validated file in the hackathon. It acts as the LLM Controller.

* **Root Location:** It MUST be named `inference.py` and placed in the absolute root directory.
* **OpenAI Client:** You MUST use the `openai` Python package to make LLM calls. Do not use requests, anthropic, or other clients.
* **Mandatory Variables:** The script MUST accept and use these exact environment variables:
    * `API_BASE_URL` (The API endpoint for the LLM)
    * `MODEL_NAME` (The model identifier)
    * `HF_TOKEN` (The Hugging Face / API key)
* **Mandatory Logging Format:** The script MUST emit `stdout` logs strictly using the provided OpenEnv logging format. Any deviation in field names, ordering, or formatting will result in an immediate score of 0.
    * Must use: `[START] {"task": ..., "env": ..., "model": ...}`
    * Must use: `[STEP] {"step": ..., "action": ..., "reward": ..., "done": ..., "error": ...}`
    * Must use: `[END] {"success": ..., "steps": ..., "score": ..., "rewards": [...]}`

---

## 5. DEPLOYMENT & DOCKERIZATION
* **Dockerfile:** The root directory MUST contain a working `Dockerfile`. The automated grader will build this image. If it fails to build, we fail.
* **Hugging Face Spaces:** The environment will be deployed to HF Spaces. Ensure the FastAPI app binds to `0.0.0.0` and exposes port `7860` (the standard for Hugging Face).
* **Pre-Validation:** The `/reset` and `/step` HTTP endpoints must return a HTTP `200 OK` status when pinged by the validator script. Ensure proper error handling (try/except blocks) so the server never crashes with a `500 Internal Server Error`.