<!-- Placeholder doc --># SYSTEM INSTRUCTION: OPENENV API & ARCHITECTURE SPECIFICATION

**To the AI Agent reading this:** This document defines the strict API contract and architectural patterns required to build a compliant OpenEnv environment. You MUST follow these structural rules when writing code in the `server/` and `models.py` files.

---

## 1. THE DATA BUS (`models.py`)
OpenEnv relies on strict type-safety for the LLM-to-Environment communication. 
* **Requirement:** All schemas MUST be defined using `pydantic.BaseModel`.
* **Action Space:** You MUST define an `Action` class. This is what the LLM agent sends to the environment. Use `pydantic.Field` to provide clear descriptions, bounds (e.g., `ge=0`, `le=1`), and defaults.
* **Observation Space:** You MUST define an `Observation` class. This is what the environment returns to the LLM agent. 
* **Mandatory RL Fields:** Your `Observation` class MUST explicitly include:
  * `reward: float` (The calculated reward for the current step).
  * `done: bool` (Whether the episode has reached a terminal state).
  * `error: Optional[str]` (Include this to pass feedback if the agent took an illegal action, rather than crashing).

---

## 2. THE CORE ENGINE (`server/env.py`)
This file contains the stateful mathematical logic of the environment (The Plant).

* **Inheritance:** Your main environment class MUST inherit from OpenEnv's base class (e.g., `from openenv_core.environment import Environment` or custom base if running raw FastAPI).
* **`reset(self) -> Observation`:** * MUST completely wipe all historical state, tracking variables, and fatigue levels.
  * MUST return a pristine initial `Observation` where `reward = 0.0` and `done = False`.
* **`step(self, action: Action) -> Observation`:**
  * This is the transition function. It takes the `Action`, updates the internal state, and calculates the reward.
  * **Idempotency Rule:** If `self.done == True`, subsequent calls to `step()` MUST NOT change the state further. It should just return the final observation until `reset()` is called.
* **Internal State Separation:** Keep the mathematical logic (like the Definition v3 redistribution or the Energy Sine Wave) in helper methods or isolated logic files (`server/logic/`). Do NOT clutter the `step()` function with raw math; call the helpers.

---

## 3. THE API WRAPPER (`server/app.py`)
To deploy to Hugging Face Spaces and allow the OpenEnv client to connect, the Python class must be wrapped in a FastAPI application.

* **Endpoints:** You MUST expose exactly two primary POST endpoints:
  * `@app.post("/reset", response_model=Observation)`
  * `@app.post("/step", response_model=Observation)`
* **Instantiation:** The environment class should be instantiated globally within `app.py` (e.g., `env = FlowStateEnv()`) so that the state persists between HTTP calls during an episode.
* **Port Binding:** The FastAPI application MUST bind to `0.0.0.0` and expose port `7860` (the Hugging Face Space default) when run via Uvicorn in the Dockerfile.

---

## 4. ERROR HANDLING & ROBUSTNESS (CRITICAL)
* **Never Crash:** If the LLM agent sends an illogical action (e.g., trying to allocate 10 hours to a 5-hour goal), the environment MUST NOT throw a Python `Exception` or return a 500 error. 
* **Graceful Degradation:** Instead, the environment should catch the illegal action, leave the state unchanged, set `reward = -1.0` (or a harsh penalty), populate the `error` string in the `Observation`, and return a 200 OK HTTP status. The LLM must be allowed to learn from its mistake.

---

## 5. REPRODUCIBILITY (THE SEED)
* RL environments must be deterministic for testing. If you use random numbers (e.g., to generate unexpected External Block interruptions), you MUST allow a `seed` to be passed during `__init__` or `reset()` using `random.seed()` or `numpy.random.seed()`.