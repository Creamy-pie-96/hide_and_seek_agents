from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from env.hide_seek_env import HideSeekEnv


@dataclass(frozen=True)
class LegacyCurriculumLevel:
    name: str
    width: int
    height: int
    n_food: int
    n_heavy: int
    prep_steps: int
    max_steps: int


def default_legacy_curriculum_levels() -> List[LegacyCurriculumLevel]:
    return [
        LegacyCurriculumLevel("L0_one_room", width=12, height=12, n_food=2, n_heavy=0, prep_steps=6, max_steps=120),
        LegacyCurriculumLevel("L1_small", width=24, height=24, n_food=6, n_heavy=1, prep_steps=12, max_steps=220),
        LegacyCurriculumLevel("L2_medium", width=30, height=30, n_food=10, n_heavy=2, prep_steps=16, max_steps=300),
        LegacyCurriculumLevel("L3_full", width=36, height=36, n_food=14, n_heavy=4, prep_steps=20, max_steps=400),
    ]


class LegacySeekerCurriculumManager:
    """Promotes map complexity when seeker performance is stable enough."""

    def __init__(
        self,
        base_seed: int | None,
        levels: List[LegacyCurriculumLevel] | None = None,
        min_rollouts_per_level: int = 60,
        eval_window: int = 20,
        promote_if_hiders_caught_mean_at_least: float = 1.2,
        promote_if_seeker_return_mean_at_least: float = -20.0,
    ):
        self.base_seed = base_seed
        self.levels = levels or default_legacy_curriculum_levels()
        self.min_rollouts_per_level = int(min_rollouts_per_level)
        self.eval_window = int(eval_window)
        self.promote_if_hiders_caught_mean_at_least = float(promote_if_hiders_caught_mean_at_least)
        self.promote_if_seeker_return_mean_at_least = float(promote_if_seeker_return_mean_at_least)

        self.level_idx = 0
        self._last_applied_level_idx = -1
        self._rollouts_since_level_start = 0
        self._recent_s_return: List[float] = []
        self._recent_hiders_caught: List[float] = []

    @property
    def current_level(self) -> LegacyCurriculumLevel:
        return self.levels[self.level_idx]

    def _build_env(self, level: LegacyCurriculumLevel, seed: int | None) -> HideSeekEnv:
        return HideSeekEnv(
            width=level.width,
            height=level.height,
            n_food=level.n_food,
            n_heavy=level.n_heavy,
            prep_steps=level.prep_steps,
            max_steps=level.max_steps,
            seed=seed,
        )

    def _apply_level(self, trainer, level_idx: int) -> None:
        self.level_idx = level_idx
        level = self.current_level
        trainer.env = self._build_env(level, self.base_seed)
        trainer._cached_obs = None
        trainer._reset_all_hidden()
        self._rollouts_since_level_start = 0
        self._recent_s_return.clear()
        self._recent_hiders_caught.clear()
        self._last_applied_level_idx = level_idx
        print(
            f"[curriculum] switched to {level.name} "
            f"(w={level.width},h={level.height},food={level.n_food},heavy={level.n_heavy},prep={level.prep_steps})"
        )

    def on_rollout_start(self, trainer, rollout_idx: int) -> None:
        if self._last_applied_level_idx != self.level_idx:
            self._apply_level(trainer, self.level_idx)

    def on_rollout_end(self, trainer, rollout_idx: int, stats: dict) -> None:
        self._rollouts_since_level_start += 1

        episodes = stats.get("episodes", []) or []
        for ep in episodes:
            self._recent_s_return.append(float(ep.get("ep_s_return", 0.0)))
            self._recent_hiders_caught.append(float(ep.get("hiders_caught", 0.0)))

        self._recent_s_return = self._recent_s_return[-self.eval_window:]
        self._recent_hiders_caught = self._recent_hiders_caught[-self.eval_window:]

        if self.level_idx >= len(self.levels) - 1:
            return
        if self._rollouts_since_level_start < self.min_rollouts_per_level:
            return
        if not self._recent_s_return:
            return

        s_ret_mean = float(np.mean(self._recent_s_return))
        h_caught_mean = float(np.mean(self._recent_hiders_caught)) if self._recent_hiders_caught else 0.0

        if (
            s_ret_mean >= self.promote_if_seeker_return_mean_at_least
            and h_caught_mean >= self.promote_if_hiders_caught_mean_at_least
        ):
            self._apply_level(trainer, self.level_idx + 1)
