# FIX REPORT — hide_and_seek_agents

Date: 2026-03-29
Scope: Full stabilization and refactor pipeline based on completed audit.

---

## Critical Issues

### C1 — Missing core object module (`env/objects.py` is empty)

**Explanation**: The project design specifies object mechanics in a dedicated module, but the file is empty.
**Impact**: Object lifecycle is ad-hoc and tightly coupled to world internals; extension and testing are blocked.
**Fix approach**: Implement typed object models (food, fake food, scent, barricade, light switches, heavy objects) and central object-state helpers used by world/env.

### C2 — Missing rollout-memory module (`rl/memory.py` is empty)

**Explanation**: Rollout data is embedded in trainer instead of a standalone memory component.
**Impact**: Poor separation of concerns, fragile training code, difficult masking/batching correctness.
**Fix approach**: Implement a dedicated memory API for transition storage, alive masks, tensor conversion, and GAE-ready batch extraction.

### C3 — Coordinated sweep logic is incorrect

**Explanation**: Sweep currently triggers when seekers are in same room/positions, not based on same-step distinct-door entry.
**Impact**: Repeated unintended stuns and reward exploitation; game mechanics invalid.
**Fix approach**: Track previous and current positions per seeker and enforce same-step room entry through distinct door tiles.

### C4 — Scent is not observable in agent observations

**Explanation**: Scent exists in `scent_map` but observation encodes only tile IDs.
**Impact**: Mechanic cannot be learned or used by policies.
**Fix approach**: Add scent channel in observation generation and include it in network input parsing.

### C5 — Dead-agent transitions are incorrectly added into PPO batches

**Explanation**: Trainer writes dummy transitions for dead agents for remaining rollout steps.
**Impact**: Corrupted policy/value updates and unstable training.
**Fix approach**: Store alive masks; exclude dead-agent timesteps from optimization and GAE computation.

### C6 — Invalid actions can crash (`Action(action)` cast)

**Explanation**: Out-of-range actions can throw exceptions.
**Impact**: Runtime crashes under noisy policy/output corruption.
**Fix approach**: Validate and clamp/fallback invalid actions in environment before dispatch.

### C7 — Spawn partition can fail on low room counts

**Explanation**: Room splitting assumes both hider and seeker room groups are non-empty.
**Impact**: `random.choice` failures on edge maps.
**Fix approach**: Add robust spawn fallback ensuring non-empty pools and safe placement.

---

## Major Issues

### M1 — Centralized critic claim is inaccurate

**Explanation**: Documentation says centralized value function; implementation is decentralized value.
**Impact**: Misleading architecture and experiment interpretation.
**Fix approach**: Implement centralized value input (team context) in network/trainer and align docs.

### M2 — Recurrent training mismatch (LSTM used in inference, not properly in PPO update)

**Explanation**: Hidden state is used for action selection but discarded in updates.
**Impact**: Train/inference mismatch and unstable learning.
**Fix approach**: Move to consistent feed-forward policy for now (or full sequence training). Adopt feed-forward for correctness/stability.

### M3 — Full blackout team condition incomplete

**Explanation**: Does not enforce “different rooms” and time-window coordination.
**Impact**: Mechanic can be gamed and diverges from design.
**Fix approach**: Require all three hiders on switches in distinct rooms with synchronized signal/toggle event.

### M4 — Light toggle risk mechanic (location reveal) is missing

**Explanation**: Turning off lights lacks seeker-visible reveal signal.
**Impact**: Intended risk/reward tradeoff absent.
**Fix approach**: Add reveal ping with TTL and include in observations/render state.

### M5 — Renderer performs O(H*W*R) room lookup per frame

**Explanation**: Per-tile iteration scans all rooms.
**Impact**: Avoidable frame-time cost at larger maps.
**Fix approach**: Provide tile-to-room map in render state and use O(1) lookup.

### M6 — No tests/CI coverage

**Explanation**: No deterministic/unit integration checks.
**Impact**: Refactors can silently break mechanics.
**Fix approach**: Add unit tests for world/env invariants and a basic CI workflow.

### M7 — Documentation is insufficient (`README.md` minimal)

**Explanation**: No setup, usage, architecture, or troubleshooting guidance.
**Impact**: Onboarding and reproducibility poor.
**Fix approach**: Expand README with setup, run commands, mechanics, and validation steps.

### M8 — Missing configuration architecture

**Explanation**: Hyperparameters and environment knobs are scattered constants.
**Impact**: Hard to run reproducible experiments and sweeps.
**Fix approach**: Introduce structured config dataclasses and CLI wiring.

### M9 — Relay vision mechanic missing

**Explanation**: Team mechanic discussed in design not implemented.
**Impact**: Feature parity gap and weaker cooperative seeker behavior.
**Fix approach**: Add seeker adjacency relay visibility rule in vision/catch logic.

### M10 — API claim “PettingZoo-style” is ambiguous/inaccurate

**Explanation**: Interface is custom dict-based, not true PettingZoo API.
**Impact**: Integration expectations may be wrong.
**Fix approach**: Clarify API as custom parallel multi-agent interface in docs/comments.

### M11 — Unsafe checkpoint loading with `torch.load`

**Explanation**: Untrusted pickle payload risk.
**Impact**: Security vulnerability in loading paths.
**Fix approach**: Enforce trusted-only warning + safer loading parameters where supported + schema validation.

### M12 — Reproducibility controls are incomplete

**Explanation**: Global seeding and deterministic backend configuration are missing/inconsistent.
**Impact**: Non-reproducible experiments.
**Fix approach**: Centralize deterministic seed setup for Python/NumPy/Torch and log seed in checkpoint.

### M13 — Numerical stability guards missing

**Explanation**: No NaN/Inf checks on loss/advantages.
**Impact**: Silent divergence and wasted compute.
**Fix approach**: Add finite checks with fail-fast diagnostics and optional gradient-skip behavior.

---

## Minor Issues

### m1 — Renderer HUD has dead/duplicated text path

**Explanation**: One loop draws then breaks; another loop does actual layout.
**Impact**: Confusing and harder to maintain.
**Fix approach**: Remove dead path and keep a single explicit HUD draw flow.

### m2 — Naming and action terminology inconsistencies

**Explanation**: Mixed naming (`BARRICADE_DOOR` vs `BARRICADE`, etc.).
**Impact**: Cognitive overhead and mistakes during maintenance.
**Fix approach**: Normalize naming across comments/constants/docs.

### m3 — Weak type annotations in core env/trainer code

**Explanation**: Generic dict types and optional handling warnings.
**Impact**: Lower static safety and readability.
**Fix approach**: Tighten typing and add explicit non-None assertions/helpers.

---

## Execution Order

1. Resolve all Critical issues C1 → C7.
2. Resolve all Major issues M1 → M13.
3. Resolve all Minor issues m1 → m3.
4. Run diagnostics and validation checks after each severity tier.
