import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim2.core import PrimitiveHideSeekSim
from sim2.entities import Action, Team, Tile


class TestSim2Core(unittest.TestCase):
    def test_reset_is_deterministic(self):
        a = PrimitiveHideSeekSim(width=20, height=20)
        b = PrimitiveHideSeekSim(width=20, height=20)
        sa = a.reset(seed=123)
        sb = b.reset(seed=123)

        self.assertEqual(sa["width"], sb["width"])
        self.assertEqual(sa["height"], sb["height"])
        self.assertEqual(sa["agents"], sb["agents"])
        self.assertEqual(sa["grid"].tolist(), sb["grid"].tolist())

    def test_prep_freezes_seekers(self):
        sim = PrimitiveHideSeekSim(width=20, height=20, prep_steps=3)
        sim.reset(seed=7)
        before = {a["id"]: (a["row"], a["col"]) for a in sim.get_state()["agents"] if a["team"] == int(Team.SEEKER)}

        actions = {i: int(Action.LEFT) for i in range(6)}
        sim.step(actions)
        after = {a["id"]: (a["row"], a["col"]) for a in sim.get_state()["agents"] if a["team"] == int(Team.SEEKER)}
        self.assertEqual(before, after)

    def test_wall_collision_blocks(self):
        sim = PrimitiveHideSeekSim(width=20, height=20, prep_steps=0)
        state = sim.reset(seed=1)
        hider0 = next(a for a in state["agents"] if a["id"] == 0)

        # Force hider near a wall and move into wall.
        sim.state.agents[0].row = 1
        sim.state.agents[0].col = 1
        sim.state.grid[1, 0] = int(Tile.WALL)

        sim.step({0: int(Action.LEFT)})
        self.assertEqual((sim.state.agents[0].row, sim.state.agents[0].col), (1, 1))

    def test_visibility_catch_works(self):
        sim = PrimitiveHideSeekSim(width=20, height=20, prep_steps=0)
        sim.reset(seed=5)

        # Put seeker facing hider with clear line.
        seeker = next(a for a in sim.state.agents if a.team == Team.SEEKER)
        hider = next(a for a in sim.state.agents if a.team == Team.HIDER)
        seeker.row, seeker.col, seeker.direction = 5, 5, (0, 1)
        hider.row, hider.col = 5, 8

        for c in range(6, 8):
            sim.state.grid[5, c] = int(Tile.EMPTY)

        sim.step({})
        self.assertFalse(hider.alive)

    def test_same_actions_same_trajectory(self):
        sim_a = PrimitiveHideSeekSim(width=20, height=20, prep_steps=1)
        sim_b = PrimitiveHideSeekSim(width=20, height=20, prep_steps=1)
        sim_a.reset(seed=99)
        sim_b.reset(seed=99)

        actions_seq = [
            {0: int(Action.RIGHT), 1: int(Action.DOWN), 2: int(Action.STAY), 3: int(Action.LEFT), 4: int(Action.UP), 5: int(Action.LEFT)},
            {0: int(Action.RIGHT), 1: int(Action.DOWN), 2: int(Action.LEFT), 3: int(Action.LEFT), 4: int(Action.UP), 5: int(Action.LEFT)},
            {0: int(Action.UP), 1: int(Action.DOWN), 2: int(Action.LEFT), 3: int(Action.RIGHT), 4: int(Action.RIGHT), 5: int(Action.LEFT)},
        ]

        for acts in actions_seq:
            sim_a.step(acts)
            sim_b.step(acts)

        self.assertEqual(sim_a.trajectory_digest(), sim_b.trajectory_digest())


if __name__ == "__main__":
    unittest.main()
