import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.hide_seek_env import HideSeekEnv
from rl.mappo import MAPPOTrainer
from rl.curriculum import LegacySeekerCurriculumManager


class TestLegacyCurriculum(unittest.TestCase):
    def setUp(self) -> None:
        env = HideSeekEnv(width=36, height=36, n_food=14, n_heavy=4, seed=7)
        self.trainer = MAPPOTrainer(env=env, device="cpu", render=False)

    def test_applies_initial_level_on_rollout_start(self):
        cm = LegacySeekerCurriculumManager(base_seed=7)
        cm.on_rollout_start(self.trainer, rollout_idx=0)

        self.assertEqual(self.trainer.env.gen.width, 12)
        self.assertEqual(self.trainer.env.gen.height, 12)
        self.assertEqual(self.trainer.env.gen.n_food, 2)
        self.assertEqual(self.trainer.env.gen.n_heavy_obj, 0)
        self.assertEqual(self.trainer.env.prep_steps, 6)
        self.assertEqual(self.trainer.env.max_steps, 120)

    def test_promotes_when_thresholds_met(self):
        cm = LegacySeekerCurriculumManager(
            base_seed=7,
            min_rollouts_per_level=1,
            eval_window=2,
            promote_if_hiders_caught_mean_at_least=1.0,
            promote_if_seeker_return_mean_at_least=-10.0,
        )
        cm.on_rollout_start(self.trainer, rollout_idx=0)

        good_stats = {
            "episodes": [
                {"ep_s_return": 5.0, "hiders_caught": 2},
                {"ep_s_return": 4.0, "hiders_caught": 1},
            ]
        }
        cm.on_rollout_end(self.trainer, rollout_idx=0, stats=good_stats)

        self.assertEqual(cm.level_idx, 1)
        self.assertEqual(self.trainer.env.gen.width, 24)
        self.assertEqual(self.trainer.env.gen.height, 24)


if __name__ == "__main__":
    unittest.main()
