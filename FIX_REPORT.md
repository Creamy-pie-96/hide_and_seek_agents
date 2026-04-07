# ISSUE TRACKER — hide_and_seek_agents

Date: 2026-03-30  
Scope: Open issues only (resolved items removed).

---

## Bug

### B3 — Ursina renderer may fail in headless/no-display environments

- Location: [render/renderer_ursina.py](render/renderer_ursina.py), [play.py](play.py), [train.py](train.py)
- Problem: 3D renderer requires a graphics context; startup can fail on servers/CI without display.
- Impact: 3D playback/training render path is unavailable in headless setups.
- Action:
  1. Add explicit runtime guard and clear error message with fallback suggestion (`--renderer pygame` or `--no-render`);
  2. Add a smoke command in docs for local desktop-only validation.

### B1 — Recurrent hidden state reused after policy updates (trainer correctness risk)

- Location: [rl/mappo.py](rl/mappo.py#L269), [rl/mappo.py](rl/mappo.py#L332)
- Problem: `self.hidden` persists while network weights are updated between rollouts. Hidden state was produced by older parameters and is reused with new parameters.
- Impact: unstable action selection/value estimation; training drift that is hard to diagnose.
- Action:
  1.  reset all hidden states after each PPO update block, while keeping environment continuity;
  2.  add regression test for hidden-state lifecycle.

### B2 — Replay input validation is shallow

- Location: [play.py](play.py#L86), [play.py](play.py#L114)
- Problem: only top-level replay schema and non-empty `frames` are validated; malformed frame payloads can crash mid-playback.
- Impact: brittle playback pipeline and poor error diagnostics.
- Action:
  1.  validate each frame has required keys (`grid`, `agents`, `scent_map`, dimensions);
  2.  fail early with frame index + reason.

---

## Design Flaw

### D1 — Headless path still hard-imports GUI modules

- Location: [train.py](train.py#L24), [play.py](play.py#L20)
- Problem: GUI modules are imported unconditionally at module load. `--no-render` still depends on Pygame import side effects.
- Impact: avoidable startup warnings/failures in headless environments and unnecessary dependency coupling.
- Action:
  1.  move renderer imports inside backend selection paths;
  2.  keep pure headless train/play executable without GUI packages.

### D2 — Plan/architecture drift was unmanaged

- Location: [plan.md](plan.md)
- Problem: plan previously described obsolete greenfield tasks and mismatched API assumptions.
- Impact: wrong execution priorities and wasted engineering cycles.
- Action:
  1.  keep plan synchronized with real architecture;
  2.  enforce “open issues only” planning updates per release cycle.

---

## Performance Issue

### P1 — Ursina dynamic entity updates need explicit state-diff guarantees

- Location: [render/renderer_ursina.py](render/renderer_ursina.py)
- Problem: dynamic objects are updated each frame; update logic must stay diff-based for larger maps.
- Impact: avoidable CPU/GPU churn and frame drops.
- Action:
  1. keep add/remove/update strictly keyed by tile-state diff;
  2. add unit test for unchanged-frame no-op updates.

### P2 — Metrics logger flushes on each row

- Location: [rl/monitoring.py](rl/monitoring.py#L95), [rl/monitoring.py](rl/monitoring.py#L100)
- Problem: sync flush per write.
- Impact: excessive I/O overhead for long runs.
- Action:
  1.  switch to buffered flush interval;
  2.  force flush on checkpoint/save/exit.

---

## Missing Feature

### F1 — Ursina semantic parity with environment state is incomplete

- Location: [render/renderer_ursina.py](render/renderer_ursina.py), [env/hide_seek_env.py](env/hide_seek_env.py#L548)
- Problem: Ursina view currently focuses on map primitives + agents; overlays such as room-light dimming, scent intensity and reveal ping are not yet rendered.
- Impact: 3D view can diverge from game truth and mislead debugging.
- Action:
  1. add room-light darkening overlay;
  2. add scent visualization;
  3. add light-ping marker;
  4. parity-check against Pygame frame semantics.

### F2 — No automated tests for trainer/replay/Ursina integration paths

- Location: [tests/test_env_core.py](tests/test_env_core.py)
- Problem: tests cover core env only; critical trainer/replay/renderer regressions are unguarded.
- Impact: fixes regress silently.
- Action:
  1.  add trainer smoke test (`--rollouts 1` equivalent);
  2.  add replay schema validation tests;
  3.  add renderer state-update unit tests (mock server handles).

---

## Status of previously tracked issues

Previously listed critical/major/minor items from 2026-03-29 are treated as resolved and removed from this file.
