# SYSTEM INSTRUCTION: REWARD SHAPING & SIGNAL PROCESSING

**To the AI Agent reading this:** You are tasked with implementing the `_compute_reward()` function for the "FlowState RL" environment. You MUST NOT use a simple "1 for win, 0 for loss" sparse reward. You MUST implement the following multi-objective, continuous reward formulas.

---

## 1. THE MULTI-OBJECTIVE REWARD FORMULA
Every time the `step()` function is called, the environment must calculate a scalar `reward` float. The master formula is:

`R_step = (W_p * P) + (W_a * A) + (W_c * C) - (W_o * O) - (W_f * F)`

Where the weights are: `W_p=0.4` (Productivity), `W_a=0.3` (Energy Alignment), `W_c=0.1` (Context/Rewire), `W_o=0.5` (Overload), `W_f=0.5` (Fatigue).

---

## 2. REWARD COMPONENTS EXPLAINED (THE MATH)

### A. Productivity (P) [Range: 0.0 to +1.0]
* **Logic:** Reward the agent for scheduling required `W_goal` hours without violating deadlines.
* **Formula:** `P = (Scheduled_Goal_Hours / Total_Required_Goal_Hours)` for that specific step.

### B. Energy Alignment (A) [Range: -1.0 to +1.0]
* **The Signal:** Human energy is modeled as a continuous circadian sine wave peaking at 2:00 PM (14:00) and dipping at 2:00 AM.
* **Waveform:** `E(t) = 0.5 + 0.4 * sin(2 * pi * (t - 8.0) / 24.0)` where `t` is the hour of the day (0.0 to 24.0).
* **Logic:** * If the agent schedules a **High-Energy** (Type 1 or 2) Goal when `E(t) > 0.7`, `A = +1.0`.
    * If scheduled when `E(t) < 0.4`, `A = -1.0` (Penalty for fighting the body clock).
    * If it schedules a **Break Block** during an energy dip (`E(t) < 0.4`), `A = +0.8`.

### C. Context / The R.E.A.L. Bonus (C) [Range: 0.0 or +1.0]
* **Logic:** Deep work requires mental preparation.
* **Formula:** If the agent's action inserts at least `0.1` hours of `W_break` immediately preceding a High-Energy Goal block, grant a `C = 1.0` "Rewire Bonus".

### D. Overload Penalty (O) [Range: 0.0 to +1.0]
* **Logic:** Time is strictly conserved. Overlapping fixed blocks is an absolute failure.
* **Formula:** If the agent's proposed schedule overlaps an `External Block` or causes the daily sum to exceed 24.0 hours, `O = 1.0`. (This triggers the heavy `-0.5` weighted penalty).

### E. Fatigue Penalty (F) [Range: 0.0 to +1.0]
* **Logic:** Evaluated from the environment's current `fatigue_level` state variable.
* **Formula:** `F = current_fatigue_level`. As fatigue approaches 1.0, the penalty grows continuously, forcing the agent to schedule breaks to reduce it.

---

## 3. HACKATHON COMPLIANCE: THE FINAL GRADER SCORE
While `R_step` guides the RL agent during training, the OpenEnv Hackathon requires a final deterministic evaluation score strictly clamped between `[0.0, 1.0]`.

When writing the automated graders for the Easy, Medium, and Hard tasks (often placed in a `graders.py` file or at the end of the inference script):
1.  Sum the cumulative rewards over the episode.
2.  Normalize the result: `final_score = max(0.0, min(1.0, total_episode_reward / theoretical_max_reward))`
3.  Ensure that a `fatigue_level >= 1.0` (Burnout) automatically caps the `final_score` at `0.0`.

---

## 4. DETERMINISTIC REQUIREMENT
Do NOT use random numbers in the reward calculation. For identical state and action inputs, the reward MUST compute to the exact same float value down to 4 decimal places.