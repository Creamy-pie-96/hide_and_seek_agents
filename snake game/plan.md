# Snake RL PPO Architecture Plan

## 1) Current architecture

- PPO training loop with CNN actor-critic and GAE in [snake_game/ppo_train.py](snake_game/ppo_train.py)
- Agent optimization and checkpointing in [snake_game/ppo_agent.py](snake_game/ppo_agent.py)
- Environment with optional static opponent in [snake_game/env_v2.py](snake_game/env_v2.py)
- Training/play wrappers in [train.py](train.py), [play.py](play.py)

## 2) Acceptance criteria (active)

Training considered healthy when all are met:

1. `eval_avg_score >= 1.5` by episode 500
2. `eval_avg_score >= 3.0` by episode 1500
3. non-zero score in >35% eval episodes
4. no dominant looping pattern in visual replay

If not met by episode 1500, tune curriculum/entropy/self-play parameters before scaling training budget.

---

## 3) Execution steps (implemented + next)

1. ✅ Resume from checkpoint (`--resume`) with trainer-state restore.
2. ✅ Adaptive entropy controller with bounded updates.
3. ✅ Proficiency-based curriculum manager (promotion by sustained eval).
4. ✅ Static opponent mode (`heuristic`, `last_best`) for competition pressure.
5. 🔄 Run 3-seed sweeps and compare eval-score variance.
6. 🔄 Add automated tests for resume continuity and entropy-controller behavior.

---

## 4) Default tuned PPO run template

Example target settings:

- `episodes=2500`
- `self-play=true`, `self-play-mode=heuristic` (or `last_best` after baseline)
- `use-adaptive-entropy=true`
- `use-curriculum=true`, `curriculum-promote-streak=3`
- evaluate every 25 episodes, 10 eval episodes each

---

## 5) Deliverables

- updated trainer logs with train + eval metrics
- replay files for first/best/last eval episodes
- side-by-side comparison report: baseline PPO vs self-play PPO
- final recommendation with reproducible command set
