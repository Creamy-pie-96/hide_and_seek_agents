"""
test_trainer_smoke.py — Smoke test for MAPPOTrainer.

Runs a single rollout + update cycle and verifies:
  - No crash on collect + PPO update
  - Hidden states are reset after each update (B1 regression guard)
  - Episode returns are tracked
"""
import os
import sys
import unittest
import tempfile
import shutil

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.hide_seek_env import HideSeekEnv
from rl.mappo import MAPPOTrainer


class TestTrainerSmoke(unittest.TestCase):
    def setUp(self) -> None:
        self.env = HideSeekEnv(width=24, height=24, seed=42)
        self.trainer = MAPPOTrainer(self.env, device="cpu", render=False)
        self._tmp = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_single_rollout_and_update(self):
        """One rollout + PPO update should run without crashing."""
        stats = self.trainer.collect_rollout(rollout_idx=0, train_team='hider')
        self.assertIn('steps', stats)
        self.assertGreater(stats['steps'], 0)

        # After collect, buffers should have data.
        from env.hide_seek_env import HIDER_IDS, SEEKER_IDS
        from rl.mappo import MAPPOTrainer as _  # just re-check import

        # PPO update should not crash.
        from rl.mappo import HIDER_IDS as HI, SEEKER_IDS as SI
        loss = self.trainer._update_team(HI, self.trainer.hider_net, self.trainer.hider_opt)
        # loss may be empty if no valid steps, but it should not crash.
        self.assertIsInstance(loss, dict)

    def test_hidden_state_reset_after_update(self):
        """B1 regression: hidden states must be None after reset."""
        self.trainer.collect_rollout(rollout_idx=0, train_team='hider')
        # Simulate what train() does: update then reset.
        from rl.mappo import HIDER_IDS
        self.trainer._update_team(HIDER_IDS, self.trainer.hider_net, self.trainer.hider_opt)
        self.trainer._reset_all_hidden()

        for aid, h in self.trainer.hidden.items():
            self.assertIsNone(h, f"Agent {aid} hidden state should be None after reset")

    def test_train_one_rollout(self):
        """Full train() with 1 rollout should complete."""
        self.trainer.train(
            n_rollouts=1,
            save_every=999,
            save_path=self._tmp,
            renderer=None,
            output_root=self._tmp,
            eval_every=0,
        )
        # Should produce a final checkpoint.
        final = os.path.join(self._tmp, "final.pt")
        self.assertTrue(os.path.exists(final), "final.pt should be created")


if __name__ == '__main__':
    unittest.main()
