# Q-Learning for Mastermind — Complete Study & Implementation Guide

---

## Table of Contents

1. [Assignment Overview](#1-assignment-overview)
2. [Core RL Concepts You Need](#2-core-rl-concepts-you-need)
3. [Architecture of the Provided Files](#3-architecture-of-the-provided-files)
4. [How the System Works Together](#4-how-the-system-works-together)
5. [Step-by-Step Implementation Plan](#5-step-by-step-implementation-plan)
6. [The Algorithm in Detail](#6-the-algorithm-in-detail)
7. [Important Variables, Functions & Data Structures](#7-important-variables-functions--data-structures)
8. [Debugging Guide](#8-debugging-guide)
9. [Recommended Workflow](#9-recommended-workflow)
10. [Final Checklist](#10-final-checklist)

---

## 1. Assignment Overview

### What you are building

You are implementing a **tabular Q-learning agent** that learns to play Mastermind — a code-breaking game.

In Mastermind, there is a secret code (e.g., `(1, 2)` for a 2-position, 4-colour version). Each turn you guess a code, and the environment tells you:

- **Black pegs** — correct colour in the correct position.
- **White pegs** — correct colour but in the wrong position.

The agent's goal is to guess the secret code within `MAX_TURNS` attempts.

### What the RL problem looks like

| RL Concept | In This Assignment |
|---|---|
| **Agent** | The `QLearningAgent` |
| **Environment** | `MastermindEnv` |
| **State** | Tuple of recent guess/feedback records |
| **Action** | An integer index corresponding to a code to guess |
| **Reward** | `+10.0` for a win, `-1.0` for every other step |
| **Episode** | One full game (reset → guesses → win or run out of turns) |

### What you must do

1. **Implement** `select_action` and `update` in `QLearningAgent` (the two placeholder methods in `student_template.py`).
2. **Run Experiment 1** — sweep over `history_length` values and find the best one.
3. **Run Experiment 2** — sweep over `epsilon` values using the best `history_length`.
4. **Run Experiment 3** — plot learning curves for your chosen settings.
5. **Write a report** answering the four analysis questions in the template.

### Expected final behaviour

A correctly trained agent should achieve a win rate well above the random baseline (which is roughly 45–55% for 2 positions, 4 colours). A strong agent can reach 90%+ win rate with a good history length and epsilon.

---

## 2. Core RL Concepts You Need

### 2.1 The basic loop

Reinforcement learning is a trial-and-error process. At each step:

1. The agent **observes a state** (what does the world look like right now?).
2. The agent **chooses an action** (what should I do?).
3. The environment **transitions** to a new state and returns a **reward**.
4. The agent uses this experience to **improve its decisions**.

This repeats until the episode ends (win or out of turns).

### 2.2 Episodes

An **episode** is one complete game of Mastermind: from `env.reset()` until `done = True`. The agent plays many thousands of episodes during training, gradually improving.

### 2.3 States and actions

- **State** — a summary of what the agent can observe. Here, it is a tuple of recent `(guess, black, white)` records. The agent uses this to decide what to guess next.
- **Action** — an integer from `0` to `env.num_codes - 1`. Each integer maps to a specific code tuple (e.g., action `5` might map to `(1, 1)`).

### 2.4 The Q-value — what we are learning

The **Q-value**, written `Q(s, a)`, is the agent's estimate of "how good is it to take action `a` from state `s`?" More precisely, it estimates the total future reward expected if we take action `a` now and then act optimally thereafter.

At the start of training all Q-values are `0.0`. The agent updates them as it gains experience.

### 2.5 The Q-learning update rule

After taking action `a` in state `s`, receiving reward `r`, and ending up in `next_state`:

```
target = r + gamma * max_a'(Q(next_state, a'))    [if not done]
target = r                                          [if done / terminal]

Q(s, a) <-- Q(s, a) + eta * (target - Q(s, a))
```

Breaking this down:

- **`target`** — what we *wish* `Q(s, a)` was, based on what we just observed. For terminal states (the episode is over), there is no future, so the target is just `r`. For non-terminal states, we add the discounted value of the best next action.
- **`target - Q(s, a)`** — the **TD error** (temporal difference error). This is how wrong our current estimate was.
- **`eta` (learning rate)** — controls how big a step we take toward the target. `eta = 1.0` means we replace the old estimate entirely; `eta = 0.0` means we never learn. A value around `0.1`–`0.3` is typical and is fixed at `0.2` in this assignment.
- **`gamma` (discount factor)** — how much we value future rewards. `gamma = 0.99` means future rewards are almost as valuable as immediate ones. `gamma = 0.0` would make the agent completely short-sighted.

### 2.6 Exploration vs exploitation

The agent faces a dilemma:

- **Exploitation** — use the best action it currently knows (greedy). Gets rewards now but may miss better options.
- **Exploration** — try random actions. Risks lower rewards now but may discover better strategies.

We balance this with **epsilon-greedy**:

- With probability `epsilon`, take a random action (explore).
- With probability `1 - epsilon`, take the greedy action (exploit).

During **evaluation**, we always set `explore=False` to test the pure greedy policy — the agent uses what it has learned, without random exploration.

### 2.7 Why Q-values start at 0.0 and why that matters here

All Q-values start at `0.0`. Because the reward for winning is `+10.0` and for losing steps is `-1.0`, the Q-values for good paths will rise above `0.0` as the agent learns. Before many states have been visited, many actions tie at `0.0`, and the greedy tie-breaking is **random**. This means that even `epsilon=0` provides implicit exploration early in training — a useful property of this specific setup.

---

## 3. Architecture of the Provided Files

### 3.1 `environment.py` — the game world

This file contains `MastermindEnv`. You do **not** modify it.

**Key attributes:**

| Attribute | Meaning |
|---|---|
| `env.num_positions` | Positions in the code (default 2) |
| `env.num_colors` | Number of colours (default 4) |
| `env.num_codes` | Total possible codes = `num_colors ^ num_positions` |
| `env.max_turns` | Maximum guesses per game |
| `env.history_length` | How many past turns appear in the state |
| `env.done` | `True` if the episode has ended |
| `env.turn` | Current turn number (1-indexed after first step) |
| `env.history` | List of `(guess_tuple, black, white)` records |
| `env.secret` | The secret code (set by `reset()`) |

**Key methods:**

- `env.reset(secret=None)` — starts a new episode. Returns the initial state (all `None`s at first).
- `env.step(action)` — takes one step. Returns `(next_state, reward, done, info)`.
- `env.compute_feedback(guess, secret)` — returns `(black, white)`. Used internally.
- `env.action_to_code(action)` — converts integer action → code tuple.
- `env.code_to_action(code)` — converts code tuple → integer action.

**State format:**

The state is a tuple of length `history_length`. Each entry is either `None` (not yet played) or a tuple `(guess_pos1, guess_pos2, black, white)`.

Example with `history_length=4` after 2 guesses:
```
((0, 0, 0, 0), (1, 1, 1, 0), None, None)
```

**Reward scheme:**

- Win step: `+10.0`
- All other steps (including the last step of a lost game): `-1.0`

**Terminal condition:**

The episode ends when the agent guesses correctly (`black == num_positions`) or `turn >= max_turns`.

### 3.2 `qtable.py` — the value store

This file contains `QTable`. You do **not** modify it.

It is a dictionary mapping `state → numpy array of Q-values` (one per action). Unseen states return `0.0` as their default Q-value.

**Methods you will use:**

```python
Q.get(state, action)          # returns Q(state, action), default 0.0
Q.set(state, action, value)   # stores Q(state, action) = value
Q.get_max_value(state)        # returns max over all actions
Q.get_best_action(state)      # returns greedy action (random tie-breaking)
```

Important: `get_best_action` breaks ties **randomly**. Early in training, when everything is `0.0`, this means greedy selection is actually random — giving you free exploration.

### 3.3 `student_template.py` — your work

This file contains:

- **Configuration constants** — hyperparameters and experiment settings at the top.
- **`QLearningAgent`** — the class you complete. Currently, `select_action` always returns a random action, and `update` does nothing.
- **`train_episode`** — already written; calls your `select_action` and `update`.
- **Experiment functions** — `experiment_history_length`, `experiment_epsilon`, `experiment_learning_curves` — scaffolding already provided; you add plotting code.
- **`evaluate_agent`** — provided; runs the greedy policy for `num_episodes` games and returns `(win_rate, mean_turns)`.
- **`train_with_tracking`** — provided; trains an agent while recording evaluation snapshots.
- **Analysis questions** — at the bottom; answer these in your report.

---

## 4. How the System Works Together

Here is the full mental model of one training episode:

```
env.reset()
    └─> picks a random secret code
    └─> clears history and turn counter
    └─> returns initial state: (None, None, None, None)  [for history_length=4]

LOOP while not env.done:

    agent.select_action(state, explore=True)
        └─> with probability epsilon: random action (explore)
        └─> otherwise: Q.get_best_action(state) (exploit)
        └─> returns action (integer)

    env.step(action)
        └─> converts action to code tuple
        └─> computes (black, white) feedback
        └─> appends to env.history
        └─> increments env.turn
        └─> checks win / out-of-turns
        └─> computes reward (+10.0 or -1.0)
        └─> builds next_state from updated history
        └─> returns (next_state, reward, done, info)

    agent.update(state, action, reward, next_state, done)
        └─> computes target:
                if done:  target = reward
                else:     target = reward + gamma * Q.get_max_value(next_state)
        └─> computes current estimate: current = Q.get(state, action)
        └─> applies update: Q.set(state, action, current + eta * (target - current))

    state = next_state

END LOOP
```

After thousands of episodes, the Q-table stores learned values for states the agent has encountered. States that lead to winning get high Q-values; dead-end guesses get low ones.

---

## 5. Step-by-Step Implementation Plan

### Phase 1 — Read and run the starter code

**Why:** Make sure the environment actually runs before you touch anything.

**What to do:**
```bash
python student_template.py
```

You will see the smoke test run. The agent will behave randomly (because `select_action` is a placeholder). The win rate will be at or below the random baseline, and the test will print "Win rate is still near the random baseline; check your code."

This is expected. Your job is to fix that.

### Phase 2 — Understand the state

Before implementing anything, print a few states manually:

```python
from environment import MastermindEnv
env = MastermindEnv(num_positions=2, num_colors=4, max_turns=6, history_length=4)
state = env.reset()
print("Initial state:", state)
# --> (None, None, None, None)

next_state, reward, done, info = env.step(0)  # guess code 0 = (0,0)
print("After first guess:", next_state)
# --> ((0, 0, 0, 1), None, None, None)  [numbers depend on secret]
print("Reward:", reward, "Done:", done)
print("Info:", info)
```

Confirm you understand: the state is a tuple, and each element is either `None` or `(guess_pos1, guess_pos2, black, white)`. The Q-table uses the full tuple as a dictionary key.

### Phase 3 — Implement `select_action`

**Location:** `QLearningAgent.select_action` in `student_template.py`

**What it should do:** Epsilon-greedy action selection.

```python
def select_action(self, state: State, explore: bool = True) -> int:
    if explore and np.random.random() < self.epsilon:
        # Explore: pick a uniformly random action
        return int(np.random.randint(self.env.num_codes))
    else:
        # Exploit: pick the greedy best action
        return self.Q.get_best_action(state)
```

**Key points:**
- Only explore when `explore=True`. During evaluation, `explore=False` is passed, so the agent always acts greedily.
- `np.random.random()` returns a float in `[0, 1)`. Compare it to `self.epsilon`.
- `self.Q.get_best_action(state)` breaks ties randomly, so even greedy early in training can behave randomly.

**Common mistake:** Forgetting the `explore` flag. If you always explore, evaluation will be noisy and misleading.

**Test:** Run the smoke test. You should now see the agent picking non-random actions. The win rate may not have improved yet because `update` is still a placeholder.

### Phase 4 — Implement `update`

**Location:** `QLearningAgent.update` in `student_template.py`

**What it should do:** Apply the Q-learning update rule.

```python
def update(
    self,
    state: State,
    action: int,
    reward: float,
    next_state: State,
    done: bool,
):
    current_q = self.Q.get(state, action)

    if done:
        target = reward
    else:
        target = reward + self.gamma * self.Q.get_max_value(next_state)

    new_q = current_q + self.eta * (target - current_q)
    self.Q.set(state, action, new_q)
```

**What each line does:**

- `current_q` — the agent's current estimate for this state-action pair. Starts at `0.0` for unseen pairs.
- `target` — what we want `current_q` to move toward. For terminal states, the future has no value, so the target is just the reward. For non-terminal states, we add a discounted estimate of the best possible future reward.
- `new_q` — the updated value. The learning rate `eta` controls how far we move: `eta=1.0` replaces `current_q` entirely, `eta=0.0` never changes it.
- `Q.set` — stores the new value in the Q-table.

**Common mistakes:**
- Forgetting the `done` condition — always using `reward + gamma * max_next` even at terminal states will corrupt the Q-values because the episode is over and there is no next state to bootstrap from.
- Using `Q.get_best_action(next_state)` instead of `Q.get_max_value(next_state)` — you want the value of the best action, not the action index.
- Accidentally overwriting `state` instead of `next_state`.

**Test:** Run the smoke test again. With both methods implemented, the win rate should now climb during training and the final result should exceed 80%, triggering "Implementation sanity check passed!".

### Phase 5 — Add plotting to experiment functions

Each experiment function ends with a `# STUDENT TASK` block where you add a plot. Here is a template you can adapt for all three:

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(x, win_means * 100, "o-", label="Q-learning")
ax1.fill_between(x, (win_means - win_stds) * 100, (win_means + win_stds) * 100, alpha=0.2)
ax1.axhline(y=baseline_wr * 100, color="red", linestyle="--", label="Random baseline")
ax1.set_xlabel("History Length")  # or "Epsilon" or "Episodes"
ax1.set_ylabel("Win Rate (%)")
ax1.set_title("Win Rate vs History Length")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(x, turn_means, "o-", label="Q-learning")
ax2.fill_between(x, turn_means - turn_stds, turn_means + turn_stds, alpha=0.2)
ax2.axhline(y=baseline_turns, color="red", linestyle="--", label="Random baseline")
ax2.set_xlabel("History Length")
ax2.set_ylabel("Avg Turns (failed = max+1)")
ax2.set_title("Avg Turns vs History Length")
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("history_length_comparison.png", dpi=150)
plt.show()
```

### Phase 6 — Run the experiments

**Order matters:**

1. First run `experiment_history_length` with the default small range to verify things work.
2. Expand `HISTORY_LENGTHS` to include a few values (e.g., `(1, 2, 3, 4)`) and re-run.
3. Pick the best `history_length` based on win rate and stability.
4. Update `EPSILON_SWEEP_HISTORY_LENGTH` to your chosen value.
5. Run `experiment_epsilon` with a spread like `(0.0, 0.05, 0.1, 0.2, 0.3, 0.5)`.
6. Pick the best `epsilon`.
7. Update `CURVE_HISTORY_LENGTH` and `CURVE_EPSILON` and run `experiment_learning_curves`.

### Phase 7 — Write the report

Answer Q1–Q4 using the tables and plots from your experiments. The next section gives you the theory behind each answer.

---

## 6. The Algorithm in Detail

### Q-learning — how it maps to the code

Q-learning is an **off-policy** algorithm, meaning it learns the value of the **greedy** policy regardless of what actions the agent actually takes during training. This is why we can explore with `epsilon > 0` during training but still learn the optimal Q-values.

**Full update equation:**

```
Q(s, a) ← Q(s, a) + η · [r + γ · max_{a'} Q(s', a') − Q(s, a)]

where:
  s      = current state
  a      = action taken
  η      = learning rate (eta = 0.2)
  r      = reward received
  γ      = discount factor (gamma = 0.99)
  s'     = next state
  max_{a'} Q(s', a') = best Q-value in next state (= 0.0 if terminal)
```

**Why the target differs for terminal vs non-terminal:**

- **Non-terminal:** The game is still going. The best we can do from `s'` is worth `γ · max Q(s', a')`, so the target is `r + γ · max Q(s', a')`.
- **Terminal:** The game is over. There is no `s'` to reason about. The target is simply `r` (the final reward). Using `max Q(s', a')` here would be wrong — the Q-table for a terminal `s'` has no meaning because no actions can be taken.

In code:
```python
target = reward if done else reward + self.gamma * self.Q.get_max_value(next_state)
```

**What changes over time during training:**

- Early training: most Q-values are `0.0`. The agent explores widely. When it wins, the Q-value for that `(state, action)` pair gets pushed up toward `+10.0`. When it loses, the Q-value gets pushed down.
- Mid training: Q-values differentiate. Actions that often lead to wins get high Q-values; useless guesses get lower values.
- Late training: Q-values converge. The greedy policy stabilises. Win rate plateaus.

### How history length affects learning

With `history_length=1`, the state only contains the most recent guess and feedback. The agent cannot distinguish "I got (1,1) after starting with (0,0)" from "I got (1,1) after starting with (2,3)" — these look like the same state. This is a **partial observability** problem: the state is not fully Markov (the optimal action depends on the full game history, not just the last turn).

With `history_length=2`, the agent can see the last two turns and can better infer what codes are still consistent with all feedback. Each added turn makes the state more Markov but also exponentially expands the state space (more unique state tuples the Q-table must learn values for).

**The trade-off:** More history → more information → better decisions possible, but the state space grows and the agent needs more training to cover it. There is a sweet spot where the extra information pays off before the state space becomes unmanageable within the training budget.

---

## 7. Important Variables, Functions & Data Structures

### Hyperparameters

| Name | Value | What it does | Effect of changing |
|---|---|---|---|
| `ETA` | `0.2` | Learning rate | Higher → learns faster but less stable |
| `GAMMA` | `0.99` | Discount factor | Lower → agent is more short-sighted |
| `epsilon` | Varies | Exploration rate | Higher → more random; lower → more greedy |
| `history_length` | Varies | State memory depth | Higher → richer state; larger Q-table |
| `max_turns` | `6` | Max guesses | Fixed by the game |

### Key functions reference

| Function | Where | What it does |
|---|---|---|
| `env.reset()` | `environment.py` | Start a new episode, return initial state |
| `env.step(action)` | `environment.py` | Take action, return `(next_state, reward, done, info)` |
| `agent.select_action(state, explore)` | `student_template.py` | Epsilon-greedy action selection — **you implement** |
| `agent.update(...)` | `student_template.py` | Q-learning update — **you implement** |
| `agent.train_episode()` | `student_template.py` | One full episode of training — provided |
| `evaluate_agent(agent, n)` | `student_template.py` | Greedy evaluation over `n` games — provided |
| `train_with_tracking(...)` | `student_template.py` | Train + record snapshots — provided |
| `compute_random_baseline()` | `student_template.py` | Evaluate a random policy — provided |

### State representation detail

The state is a Python **tuple** used as a dictionary key in the Q-table. A state with `history_length=2` after one guess might be:

```python
((0, 1, 0, 0), None)
# meaning: first guess was (0,1), got black=0, white=0; second slot not yet used
```

After two guesses:
```python
((0, 1, 0, 0), (2, 3, 1, 0))
# meaning: first guess (0,1) → 0 black, 0 white
#          second guess (2,3) → 1 black, 0 white
```

Each tuple is hashable, so it works as a Q-table key directly.

### Average-turns metric

The evaluation metric for turns penalises losses:

```
average_turns = mean([actual_turns if won else max_turns + 1 for each game])
```

This is better than only averaging wins because it captures both success rate and efficiency in a single number. An agent that always loses would score `max_turns + 1 = 7`; an agent that always wins on turn 1 would score `1`. The random baseline is somewhere in between.

---

## 8. Debugging Guide

### Symptom: win rate stays near random baseline after training

**Likely cause 1:** `update` is not doing anything (still the placeholder `pass`).
- Check: print `self.Q.q` after 100 episodes — it should not be empty.

**Likely cause 2:** `update` is being called but the target is wrong.
- Check: add a print inside `update` for the first 5 calls and verify the target makes sense. A win should produce target `= 10.0`; a non-terminal step with `max_next = 0.0` should produce target `= -1.0`.

**Likely cause 3:** `select_action` always returns random (still the placeholder).
- Check: print `explore` and confirm the greedy path is taken when `explore=False`.

### Symptom: win rate is always exactly 0% or 100%

- `0%` — the agent may be learning to always pick a bad action. Check for a bug where Q-values decrease indefinitely or the update has the sign wrong.
- `100%` — unlikely to be a bug, but double-check that `explore=False` during evaluation is correct.

### Symptom: Q-values are all exactly 0.0 after training

The Q-table is not being written to. Check that `Q.set` is being called in `update`.

### Symptom: training is very slow

Reduce `NUM_COLORS` from 4 to 3 (state this in your report). You can also reduce `SWEEP_RUNS` for initial debugging.

### How to inspect Q-values

```python
# After training
agent, history = train_with_tracking(env, ...)
print("Number of states in Q-table:", len(agent.Q.q))
# Should be > 0 after training

# Check Q-values for the initial state
initial_state = env.reset()
for action in range(env.num_codes):
    print(f"Action {action} ({env.action_to_code(action)}): Q = {agent.Q.get(initial_state, action):.3f}")
```

### How to verify convergence

Plot the learning curve. A converged agent shows a plateau in win rate. If it is still rising at the end of training, you may want more training episodes.

---

## 9. Recommended Workflow

Follow this order to avoid getting overwhelmed:

**Step 1** — Read `environment.py` top-to-bottom. Print a few states manually. (15 min)

**Step 2** — Read `qtable.py`. Call `Q.get`, `Q.set`, `Q.get_max_value`, `Q.get_best_action` in a scratch script to see what they return. (10 min)

**Step 3** — Implement `select_action`. Run the smoke test. (10 min)

**Step 4** — Implement `update`. Run the smoke test. Verify "sanity check passed". (20 min)

**Step 5** — Add plotting to `experiment_history_length`. Run it with just `HISTORY_LENGTHS = (1, 2)` to verify the plot appears. (15 min)

**Step 6** — Expand `HISTORY_LENGTHS` to `(1, 2, 3, 4)` and increase `SWEEP_RUNS` to ~20. Run the full history experiment. Record which `history_length` wins. (training time varies)

**Step 7** — Update `EPSILON_SWEEP_HISTORY_LENGTH` to your chosen value. Run `experiment_epsilon` with a spread of epsilon values. (training time varies)

**Step 8** — Update `CURVE_HISTORY_LENGTH` and `CURVE_EPSILON`. Run `experiment_learning_curves`. (training time varies)

**Step 9** — Write your report using the plots and tables. Answer Q1–Q4.

**Milestones:**
- After Step 4: smoke test passes with >80% win rate ✓
- After Step 6: you know the best `history_length` ✓
- After Step 7: you know the best `epsilon` ✓
- After Step 8: you have all the plots you need ✓

---

## 10. Final Checklist

### Code

- [ ] `select_action` implements epsilon-greedy correctly (respects `explore` flag)
- [ ] `update` computes target differently for terminal vs non-terminal states
- [ ] `update` uses `eta`, `gamma`, `Q.get`, `Q.get_max_value`, `Q.set` correctly
- [ ] Smoke test passes (>80% win rate printed)
- [ ] Plotting code added to all three experiment functions
- [ ] Experiment functions produce output (tables + plots)

### Experiments

- [ ] History lengths tested include `1` and at least 2-3 larger values
- [ ] Epsilon sweep includes `0.0` and a spread across `[0, 0.6]`
- [ ] Each experiment uses multiple independent runs (at least 10–20 for final results)
- [ ] All comparisons include a random baseline computed under the same configuration
- [ ] Final learning curves show mean ± std across multiple runs

### Report

- [ ] Q1a: explain the "target" and why it differs for terminal vs non-terminal states
- [ ] Q1b: explain the role of `eta`
- [ ] Q1c: explain the role of `gamma`
- [ ] Q2a: win rate for `history_length=1`, compared to random baseline, with explanation
- [ ] Q2b: optimal history length identified, with explanation (Markov property discussed)
- [ ] Q2c: state space growth vs training budget trade-off discussed
- [ ] Q3a: win rate for `epsilon=0`, explained using initial Q-values and tie-breaking
- [ ] Q3b: best epsilon value/regime identified, convergence/stability discussed
- [ ] Q3c: exploration-exploitation trade-off explained; why `explore=False` at evaluation
- [ ] Q4a: episodes to reach good performance; plateau behaviour discussed
- [ ] Q4b: importance of multiple runs; stability of conclusions discussed
- [ ] Q4c: why `max_turns + 1` penalty is better than averaging only wins
- [ ] Configuration clearly stated (NUM_COLORS, parameter ranges, final choices)
- [ ] Plots included with axis labels, titles, legends, and baseline reference lines

### Common traps to avoid

- Do not use `Q.get_best_action` in the `update` method — you want `Q.get_max_value`.
- Do not apply the `+ gamma * max_next` term when `done=True`.
- Do not forget to pass `explore=False` during evaluation (the provided `evaluate_agent` already does this — do not break it).
- Do not report results from a single run — always use multiple runs and report mean ± std.
- State your `NUM_COLORS` setting clearly in the report if you change it from 4 to 3.