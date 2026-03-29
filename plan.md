# Hide and Seek — Project Plan

Version: 0.1
Date: 2026-03-29

## Overview

This project is a multi-agent competitive-cooperative Hide and Seek environment implemented in Python with PyTorch for RL and Pygame for visualization. The core idea: two teams (3v3) — Hiders and Seekers — compete in procedurally generated room layouts. Agents can move, use tools (barricades, fake food, scent), and trigger team-only mechanics (heavy barricade, blackout, coordinated sweep). The environment is designed to encourage emergent strategies and deceptive behaviors.

## Goals

- Implement a PettingZoo-style multi-agent environment to support training and evaluation.
- Provide a real-time renderer for debugging and human play.
- Train agents with MAPPO (shared-weight actor-critic) in PyTorch.
- Facilitate reproducible experiments and ablation studies.

## High-level architecture / files

Project root: hide_and_seek/

- env/
  - `world.py` — procedural room generator, room graph, door/zone placement, tile map, light regions
  - `objects.py` — definitions and rules for Food, Barricade (light/heavy), LightSwitch, ScentTrail, FakeFood; interaction logic
  - `agent.py` — Agent state model, FOV calculation, action implementation, team signals
  - `hide_seek_env.py` — PettingZoo-style multi-agent wrapper (reset, step, render hooks, obs/action spaces)

- rl/
  - `network.py` — shared actor-critic network (CNN over FOV, MLP over scalars, LSTM, policy + value heads)
  - `memory.py` — rollout buffer supporting multi-agent, GAE, advantage calculation, minibatch indexing
  - `mappo.py` — MAPPO trainer loop (collect rollouts, compute losses, gradients, optimization, KL/clip or centralized value)

- render/
  - `pygame_render.py` — visualization utilities: tilemap rendering, agent sprites, FOV cone, lights, debug overlays

- `train.py` — entrypoint to run training jobs with config, logging, checkpointing
- `play.py` — load checkpoint and run interactive matches for human observation; include evaluation metrics and automated rollout scripts.
- `plan.md` — this file

## Core Mechanics (detailed)

Environment space

- Discrete 2D tile grid composed of rooms connected by doors; each room has a light state (on/off) and may include objects.
- Procedural generation ensures varied layouts per episode: variable number of rooms (e.g., 6–12), junctions, door positions.

Agent mechanics

- 3 agents per team (Hiders, Seekers). Agents have: position, direction, health (alive/dead), food count, team id.
- Local FOV: a square grid centered on agent (default 7×7) affected by room light state and possible obstructions.
- Actions (discrete):
  - Movement: up/down/left/right/stay (5)
  - Use tool: block door, drop fake food, drop scent, toggle light (4)
  - Team signal actions: push barricade, coordinated sweep trigger (2)
  - Combined discrete action space will be represented as a single categorical action (expandable) or modular multi-discrete encoding.

Objects & tools

- Food: consumable, gives reward to whoever eats it.
- Fake food: decoy that gives reward to hiders if seeker eats it.
- Scent trails: short-lived markers that influence seeker heuristics and can mislead.
- Barricades: can be light (single agent) or heavy (team-only: needs 2 agents cooperating), block doors.
- Light switches: toggle room light. Turning off lights reduces FOV but has a strategic info leak (see rule below).

Team-only mechanics

- Heavy barricade: requires two hiders adjacent to push a large block across a doorway fully.
- Signal blackout: requires all 3 hiders to toggle separate room switches within a short window to create floor blackout.
- Coordinated sweep (seekers): two seekers entering from different doors simultaneously can stun hidden agents in that room.
- Relay vision: two adjacent seekers can merge FOV to see through walls between them.

Tradeoffs

- Turning off lights reduces visibility but may reveal hider location (mechanic: toggling off triggers a short-location ping to nearby seekers or logs to team). This creates risk/reward.

## Observation space (per-agent)

- Local FOV grid (7×7) flattened — tiles annotated with objects and light-modulated visibility.
- Scalar vectors: own pos (x,y), direction (one-hot), health, food count.
- Teammate states: relative positions and basic state (alive/health/food); optionally shared via communication channel.
- Active scent trails visible within FOV.
- Light status of visible rooms.

Representation choices

- FOV channels: walkable, wall, door, food, fake_food, scent, barricade, light_on flag per tile (binary channels).

## Action space

- Discrete actions encoded as a single categorical action, mapping to movement/tool/team signals. Use a few reserved actions for team-signal coordination.

## Reward shaping

Per-episode event rewards (example scaling):

| Event                             |             Hiders |    Seekers |
| --------------------------------- | -----------------: | ---------: |
| Hider caught                      |               -100 |       +150 |
| Timer runs out (each alive hider) |                +50 |        -80 |
| Food eaten                        |                +10 |        +10 |
| Seeker enters empty room          |                  0 |         -5 |
| Fake food eaten by seeker         |                +20 |        -20 |
| Blackout executed                 |         +30 (team) | -15 (team) |
| Coordinated sweep success         | -40 (caught hider) | +60 (team) |

Notes: Play with scalars during experiments. Use per-step shaping for exploration (small penalty for unnecessary movement, small reward for team coordination signals) but avoid dominating the sparse high-level events.

## Network architecture

- Input 1: FOV grid as multi-channel image → small CNN (2–4 conv layers) → spatial embedding.
- Input 2: Scalar observations (pos, dir, health, teammate summaries) → MLP.
- Concatenate embeddings → LSTM (hidden size 256) to capture temporal dependencies.
- Two heads:
  - Policy head: MLP → softmax over discrete actions
  - Value head: MLP → scalar value estimate

- Shared weights across agents (centralized value optional: takes global state or extra features for value net).

## Training algorithm: MAPPO

- Use centralized training with decentralized execution (CTDE). Policy networks share weights; value head optionally receives global state for stability.
- Use PPO-style objective (either clipped or KL-penalized), GAE for advantage estimation, minibatch updates, multiple epochs per rollout.
- Support for recurrent policies: store and handle LSTM states in rollout buffer.

## Implementation order / milestones

1. World generator (`env/world.py`) + object primitives (`env/objects.py`) — deterministic mode and random seeds. (Milestone 1)
2. Pygame renderer (`render/pygame_render.py`) — visual debug of map, lights, objects. (Milestone 2)
3. Agent model (`env/agent.py`) + basic single-agent mechanics and movement. (Milestone 3)
4. Multi-agent env wrapper (`env/hide_seek_env.py`) — step/reset, obs/action spaces, reward events, done flags, team signals, and basic random policy baseline. (Milestone 4)
5. RL primitives (`rl/network.py`, `rl/memory.py`) — build the actor-critic and buffer with recurrent support. (Milestone 5)
6. MAPPO trainer (`rl/mappo.py`) — training loop, logging, checkpointing. (Milestone 6)
7. `train.py` and `play.py` + experiments and hyperparam sweeps. (Milestone 7)

Estimate: ~800–1000 lines of code, across modules, initially excluding tests and docs.

## Tests & verification

- Unit tests:
  - Deterministic world generation for fixed seeds.
  - Object interactions: food consumption, barricade blocking, light toggling.
  - Env step semantics: correct observation shapes, reward triggers, done flags.
- Integration: small end-to-end episode with scripted policies verifying expected events.

## Experiments and ablation studies

- Baselines:
  - Random policy, heuristic seekers (greedy to nearest scent/food), heuristic hiders (hide in far rooms).
- Ablations:
  - Remove blackout mechanic and compare emergent behavior.
  - Turn off team-only mechanics to test necessity.
  - Vary fake-food reward scale and measure deception frequency.

Metrics to track

- Episode win rates per team
- Mean episode length
- Food collected per team
- Frequency of tool usage (barricade, fake food, blackout)
- Emergent behaviors (measured qualitatively via videos)

## Infrastructure

- Logging: TensorBoard for losses, rewards, and metrics; save periodic gameplay videos (e.g., using renderer frames).
- Checkpointing: save model and optimizer state every N updates; provide `--resume` support.

## Minimal runnable demo plan (developer workflow)

1. Run `python -m render.pygame_render` (or `python render/pygame_render.py`) to view generated maps and a couple of scripted agents.
2. Run a headless single-environment sanity check that steps `env/hide_seek_env.py` with random actions for N steps.
3. Start `train.py` with a small config (4 parallel envs, small CNN, short episodes) to smoke test training loop.

## Safety and future ideas

- Keep reward design stable to avoid pathological behaviors (e.g., agents exploiting reward by looping trivial actions). Monitor for reward hacking.
- Possible extensions: curriculum learning, centralized critic with attention over teammates, parameter-sharing variants, self-play tournaments.

## Next steps (immediate)

1. Implement `env/world.py` and `env/objects.py` (procedural generator and object interactions).
2. Add basic headless tests to validate map generation and object rules.

---

This `plan.md` captures the architecture, mechanics, implementation order, and experiment ideas discussed. Use the TODO list in the project management tool to track progress against the milestones.
