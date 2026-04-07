from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

import numpy as np

from rl.monitoring import make_run_id, prepare_artifacts, save_replay_json
from render.video_utils import render_state_to_frame, write_mp4
from sim2.core import PrimitiveHideSeekSim
from sim2.entities import Action


PolicyFn = Callable[[Dict[str, Any]], Dict[int, int]]


def random_policy(state: Dict[str, Any]) -> Dict[int, int]:
    return {
        int(a["id"]): int(np.random.randint(int(Action.STAY), int(Action.RIGHT) + 1))
        for a in state.get("agents", [])
    }


@dataclass
class Sim2RunnerConfig:
    width: int = 24
    height: int = 24
    max_steps: int = 300
    prep_steps: int = 20
    output_root: str = "outputs"
    run_id: Optional[str] = None


class Sim2RolloutRunner:
    """Headless-first rollout runner with optional pluggable renderer."""

    def __init__(self, config: Sim2RunnerConfig):
        self.cfg = config
        self.run_id = config.run_id or make_run_id("sim2")
        self.artifacts = prepare_artifacts(config.output_root, self.run_id)

        self.sim = PrimitiveHideSeekSim(
            width=config.width,
            height=config.height,
            max_steps=config.max_steps,
            prep_steps=config.prep_steps,
        )

    def run_episode(
        self,
        seed: int,
        policy: PolicyFn = random_policy,
        renderer: Optional[Any] = None,
        fps: int = 20,
        record_video: bool = True,
        record_replay: bool = True,
        prefix: str = "ep",
    ) -> Dict[str, Any]:
        state = self.sim.reset(seed=seed)
        done = {"__all__": False}

        frames: List[np.ndarray] = []
        replay_frames: List[Dict[str, Any]] = []
        steps = 0
        total_rewards = {i: 0.0 for i in range(6)}

        while not done["__all__"]:
            actions = policy(state)
            state, rewards, done, info = self.sim.step(actions)
            steps += 1
            for k, v in rewards.items():
                total_rewards[int(k)] += float(v)

            render_state = self.sim.get_render_state()
            if renderer is not None:
                if steps <= 3:
                    print(f"[sim2-runner] render_step={steps} calling renderer.draw()")
                alive = renderer.draw(render_state, fps=fps)
                if not alive:
                    break

            if record_video:
                frames.append(render_state_to_frame(render_state, tile_px=8, hud_px=20))
            if record_replay:
                replay_frames.append(self.sim.get_serializable_render_state())

        video_path = None
        replay_path = None
        if record_video and frames:
            video_path = self.artifacts.eval_video_dir / f"{prefix}_{seed}.mp4"
            write_mp4(video_path, frames, fps=max(1, fps))

        if record_replay and replay_frames:
            replay_path = self.artifacts.replay_dir / f"{prefix}_{seed}.json"
            save_replay_json(
                replay_path,
                {
                    "schema_version": 1,
                    "run_id": self.run_id,
                    "seed": int(seed),
                    "frames": replay_frames,
                },
            )

        return {
            "run_id": self.run_id,
            "seed": int(seed),
            "steps": int(steps),
            "total_rewards": total_rewards,
            "alive_hiders": int(info.get("alive_hiders", 0)) if isinstance(info, dict) else 0,
            "alive_seekers": int(info.get("alive_seekers", 0)) if isinstance(info, dict) else 0,
            "video_path": None if video_path is None else str(video_path),
            "replay_path": None if replay_path is None else str(replay_path),
        }
