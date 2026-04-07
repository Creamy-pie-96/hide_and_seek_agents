# Hide and Seek — OpenAI-Style Rebuild Plan (Model-Free)

Version: 2.0  
Date: 2026-04-01

## 0) Product goal

Rebuild the game loop into a **clean, minimal, high-signal simulator** (Pac-Man-level readability), with:

- deterministic object-centric simulation,
- model-free visuals (primitives only),
- strict sim/render separation,
- preserved replay + video history for training progress.

No FBX dependency in primary runtime path.

---

## 1) Principles (hard constraints)

1. **Single source of truth**: simulation state only in sim core.
2. **Renderer is read-only**: rendering never mutates simulation.
3. **Headless-first**: training/eval do not require GUI.
4. **Deterministic by default**: seed + action log must replay exactly.
5. **Minimal visuals**: primitive shapes, flat colors, no texture/model noise.
6. **Backward-safe migration**: old system stays until parity checks pass.

---

## 2) Target architecture (v2)

### A. New simulation package

Add a new package (proposed):

- [sim2/state.py](sim2/state.py): entity dataclasses and world snapshot schema.
- [sim2/entities.py](sim2/entities.py): entity types and state flags.
- [sim2/worldgen.py](sim2/worldgen.py): deterministic map/layout generator.
- [sim2/rules.py](sim2/rules.py): movement, collision, LOS, interactions.
- [sim2/core.py](sim2/core.py): `reset(seed)`, `step(actions)`, `get_state()`.

### B. Renderer adapters

- [render/renderer_ursina.py](render/renderer_ursina.py): primitive viewer mode as default.
- [render/pygame_render.py](render/pygame_render.py): 2D debug parity view.
- Optional compatibility adapter for legacy env state during migration.

### C. RL adapter layer

- [env/hide_seek_env.py](env/hide_seek_env.py) becomes adapter/wrapper over sim2 core.
- Preserve current trainer-facing API: observations, masks, rewards, done, info.

### D. Replay/video pipeline

- Keep [render/video_utils.py](render/video_utils.py) as canonical video sink.
- Standardize replay frame schema for both training and playback.

---

## 3) Entity schema (minimal)

Each entity keeps only required fields:

- `id`
- `type` (`agent`, `wall`, `door`, `box`, `food`, `switch`, `hide_spot`)
- `position` (grid / continuous)
- `direction` (for LOS)
- `active`, `locked`, `alive`, `stunned` flags
- optional team tag for agents

This schema is the replay schema source.

---

## 4) Mechanics rollout (strict order)

## Phase A (M1): movement and visibility

1. Deterministic map loading/generation.
2. Agent movement + wall collision.
3. Seeker LOS cone + hider visibility check.
4. Preparation phase (seekers frozen at episode start).

Exit criteria:

- headless sim runs stable for fixed seeds,
- primitive viewer shows readable map + teams,
- deterministic replay works for Phase A.

## Phase B (M2): object interactions

1. Push / grab / lock on simple movable boxes.
2. Door and switch interactions.
3. Minimal object state transitions in logs/replays.

Exit criteria:

- object interactions validated by tests and replay audit.

## Phase C (M3): rewards + episodes

1. Team rewards and terminal conditions.
2. Basic curriculum hooks (parameterized world complexity).
3. Training smoke run + replay-to-video output.

Exit criteria:

- PPO/MAPPO smoke training passes,
- replay validator passes,
- MP4 generation preserved.

## Phase D (M4): curriculum + benchmark

1. Controlled randomization (rooms, doors, object count).
2. Stability/performance checks.
3. Benchmark scripts for regression tracking.

---

## 5) Cleanup strategy (remove old mess safely)

1. **Tag legacy paths** with deprecation comments.
2. Add dual-run adapter mode (`legacy` vs `sim2`) during transition.
3. When parity gates pass, delete dead branches and old coupling logic.
4. Keep only one canonical replay schema and one canonical step interface.

No hard deletes before parity + test pass.

---

## 6) Verification protocol (must run each milestone)

1. Compile:
   - `python -m py_compile` on changed modules.
2. Unit tests:
   - deterministic step/replay tests,
   - LOS and collision tests,
   - renderer snapshot sanity tests.
3. Smoke:
   - one headless train rollout,
   - one replay playback,
   - one MP4 generation check.
4. Determinism:
   - same seed + same actions => identical trajectory hash.

---

## 7) Immediate next implementation slice

Start with **M1 only**:

1. scaffold `sim2/` core state + step loop,
2. add primitive-only viewer mode,
3. wire replay serializer for new state,
4. run compile + deterministic smoke test.

This gives a clean base before object interactions and full reward logic.
