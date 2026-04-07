import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim2.core import PrimitiveHideSeekSim


class TestSim2RenderSchema(unittest.TestCase):
    def test_render_state_has_video_keys(self):
        sim = PrimitiveHideSeekSim(width=16, height=16)
        rs = sim.reset(seed=3)
        frame_state = sim.get_render_state()

        for key in ["grid", "scent_map", "agents", "step", "max_steps", "prep", "hiders_caught"]:
            self.assertIn(key, frame_state)

    def test_serializable_state_has_replay_keys(self):
        sim = PrimitiveHideSeekSim(width=16, height=16)
        sim.reset(seed=3)
        serial = sim.get_serializable_render_state()

        required = {"grid", "scent_map", "agents", "step", "max_steps", "prep", "hiders_caught", "width", "height"}
        self.assertTrue(required.issubset(serial.keys()))
        self.assertIsInstance(serial["grid"], list)
        self.assertIsInstance(serial["agents"], list)


if __name__ == "__main__":
    unittest.main()
