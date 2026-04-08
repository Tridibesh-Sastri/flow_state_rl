# SYSTEM INSTRUCTION: PROJECT DOMAIN LOGIC (FLOWSTATE RL)

**To the AI Agent reading this:** This document contains the proprietary domain logic for the "FlowState RL" scheduling environment. You MUST implement these mathematical rules, constraints, and formulas exactly as described when writing the `server/env.py` and `server/logic/` files.

---

## 1. THE PHYSICS OF THE ENVIRONMENT (TIME CONSERVATION)
The environment simulates a 24-hour day. Time is a conserved resource. The sum of all block widths (durations in hours) MUST exactly equal 24.0 at all times.

**The Time Equation:**
`W_constant + W_external + W_goal + W_break + W_reserve + W_free = 24.0`

* **Constant Block (`W_constant`):** Fixed daily survival tasks (Sleep, Food). Typically ~10 hours. Cannot be modified by the agent.
* **External Block (`W_external`):** Uncontrollable fixed events (Classes, Meetings). Cannot be modified by the agent.
* **Goal Block (`W_goal`):** Productive work. Limit to max 8 hours/day to prevent burnout.
* **Break Block (`W_break`):** Recovery cycles. Agent uses this to reduce fatigue.
* **Reserve Block (`W_reserve`):** Emergency buffer.
* **Free Time Block (`W_free`):** The mathematical remainder of the day.

---

## 2. THE 5-TIER GOAL MATRIX
The agent will interact with a list of "Goals". Goals dictate how the agent is allowed to manipulate `W_goal`. Each goal must be classified into one of 5 types:

| Goal Type | Width Compression | Deadline | Agent's Permitted Actions |
| :--- | :--- | :--- | :--- |
| **Type 1** | ❌ Not Allowed | 🔒 Fixed | **None.** Must be scheduled exactly as sized before the deadline. |
| **Type 2** | ❌ Not Allowed | 〰️ Flexible | **Shift Only.** Can delay the deadline, but cannot reduce hours. |
| **Type 3** | ❌ Not Allowed | ♾️ None | **Shift Only.** Can schedule whenever, but cannot reduce hours. |
| **Type 4** | ✅ Allowed | ♾️ None | **Compress.** Agent can squeeze the required hours to fit a busy day. |
| **Type 5** | ✅ Allowed | 〰️ Flexible | **Compress & Shift.** Full control to absorb shocks. |

---

## 3. "DEFINITION V3": VACANT TIME REDISTRIBUTION (CRITICAL ALGORITHM)
This is the core innovation of the environment. You MUST implement this logic for the transition step.

**The Rule:** If during an episode step, a Goal completes faster than its allocated `width`, the environment generates "Vacant Time".
* **Action:** The system MUST automatically distribute this Vacant Time to the remaining active goals.
* **Effect:** By subtracting the vacant time from the remaining goals' required widths, the system effectively "pulls down" (shortens) the deadlines of future goals.

---

## 4. FATIGUE & ENERGY MATH (THE CONTROL LOOP)
The state space (`Observation`) MUST track a continuous `fatigue_level` variable (bounded `[0.0, 1.0]`). This simulates human cognitive load.

**The Fatigue Transition Formula:**
Upon every `step()`, update fatigue based on the hours scheduled in the action:
`Fatigue_new = max(0.0, min(1.0, Fatigue_old + (0.10 * High_Energy_Hrs) + (0.05 * Med_Energy_Hrs) - (0.08 * Break_Hrs)))`

**Constraints & Penalties:**
* If `fatigue_level >= 1.0`, the user has burned out. The episode MUST terminate (`done = True`) and yield a massive negative reward.
* **The R.E.A.L. Rewire Action:** If the agent chooses an action to allocate `Break_Hrs` immediately before a `High_Energy_Hrs` task, it receives a "Context Switching Bonus" (positive reward) for preventing a fatigue spike.

---

## 5. INTERRUPTION SHOCK ABSORPTION (FOR HARD TASK)
When testing the environment (especially in the "Hard" difficulty task), the environment will randomly inject an unexpected `External Block` (e.g., a 2-hour emergency meeting).

**The Hierarchy of Absorption:**
The agent must dynamically recalculate the day to fit the new block. To maintain the 24-hour Time Equation, the agent MUST sacrifice flexible blocks in this exact order of priority:
1. First, consume `W_free` (Free Time).
2. If `W_free` is 0, consume `W_break` (Break Time) -> *Note: This will increase future fatigue!*
3. If `W_break` is 0, consume `W_reserve` (Reserve Buffer).
4. If `W_reserve` is 0, apply "Width Compression" to Type 4 & Type 5 Goals.