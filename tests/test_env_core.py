import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.hide_seek_env import HideSeekEnv
from env.agent import Action, OBS_DIM, FOV_SIZE, FOV_CHANNELS, GLOBAL_STATE_DIM


class TestEnvCore(unittest.TestCase):
    def setUp(self) -> None:
        self.env = HideSeekEnv(width=24, height=24, seed=123)

    def test_reset_shapes(self):
        obs, _ = self.env.reset(seed=123)
        self.assertEqual(len(obs), 6)
        for aid, arr in obs.items():
            self.assertEqual(arr.shape[0], OBS_DIM)

    def test_invalid_actions_do_not_crash(self):
        self.env.reset(seed=123)
        bad_actions = {i: 999999 for i in range(6)}
        obs, rewards, dones, info = self.env.step(bad_actions)
        self.assertEqual(len(obs), 6)
        self.assertEqual(len(rewards), 6)
        self.assertIn('__all__', dones)

    def test_scent_visible_in_observation(self):
        obs, _ = self.env.reset(seed=123)
        agent = self.env.agents[0]
        self.env.world.drop_scent(agent.row, agent.col, strength=1.0, ttl=5)

        obs2, _, _, _ = self.env.step({i: int(Action.STAY) for i in range(6)})
        vec = obs2[agent.agent_id]

        fov_len = FOV_SIZE * FOV_SIZE * FOV_CHANNELS
        fov = vec[:fov_len].reshape(FOV_SIZE, FOV_SIZE, FOV_CHANNELS)
        scent_channel = fov[:, :, 1]
        self.assertGreater(float(scent_channel.max()), 0.0)

    def test_action_masks_and_global_state_shape(self):
        self.env.reset(seed=123)
        masks = self.env.get_action_masks()
        self.assertEqual(len(masks), 6)
        for aid, m in masks.items():
            self.assertEqual(m.shape[0], self.env.action_dim)
            self.assertGreater(float(m.max()), 0.0)

        g = self.env.build_global_state()
        self.assertEqual(g.shape[0], GLOBAL_STATE_DIM)


if __name__ == '__main__':
    unittest.main()
