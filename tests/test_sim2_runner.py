import os
import sys
import unittest
import tempfile

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rl.sim2_runner import Sim2RolloutRunner, Sim2RunnerConfig


class _DummyRenderer:
    def __init__(self):
        self.calls = 0

    def draw(self, state, fps=20):
        self.calls += 1
        return True


class TestSim2Runner(unittest.TestCase):
    def test_headless_episode_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            runner = Sim2RolloutRunner(Sim2RunnerConfig(output_root=td, max_steps=40, prep_steps=3))
            stats = runner.run_episode(seed=5, renderer=None, record_video=True, record_replay=True)

            self.assertGreater(stats["steps"], 0)
            self.assertTrue(stats["video_path"] is None or os.path.exists(stats["video_path"]))
            self.assertTrue(stats["replay_path"] is None or os.path.exists(stats["replay_path"]))

    def test_pluggable_renderer_called(self):
        with tempfile.TemporaryDirectory() as td:
            runner = Sim2RolloutRunner(Sim2RunnerConfig(output_root=td, max_steps=20, prep_steps=1))
            r = _DummyRenderer()
            stats = runner.run_episode(seed=4, renderer=r, record_video=False, record_replay=False)
            self.assertGreater(r.calls, 0)
            self.assertGreater(stats["steps"], 0)


if __name__ == "__main__":
    unittest.main()
