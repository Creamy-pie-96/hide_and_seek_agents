wimport os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from render.renderer_ursina import Ursina3DRenderer


class _FakeEntity:
    def __init__(self, enabled=True):
        self.enabled = enabled


class _FakeCamera:
    def __init__(self, position=(0, 10, -10), rotation=(30, 0, 0), fov=70):
        self.position = position
        self.rotation = rotation
        self.fov = fov


class _FakeScene:
    def __init__(self, n=0):
        self.entities = [object() for _ in range(n)]


class _FakeApp:
    def __init__(self):
        self.step_calls = 0

    def step(self):
        self.step_calls += 1
        return None


class _NoStepApp:
    pass


class TestUrsinaRendererDebug(unittest.TestCase):
    def test_scene_not_empty_when_static_exists(self):
        r = Ursina3DRenderer(24, 24)
        setattr(r, "camera", _FakeCamera())
        setattr(r, "scene", _FakeScene(n=3))
        setattr(r, "app", _FakeApp())
        r._static_entities["/floor"] = _FakeEntity(enabled=True)

        snap = r.get_debug_snapshot()
        self.assertGreater(snap["tracked_total"], 0)
        self.assertGreater(snap["visible_total"], 0)
        self.assertEqual(snap["scene_entities"], 3)

    def test_enabled_entity_count(self):
        r = Ursina3DRenderer(24, 24)
        setattr(r, "camera", _FakeCamera())
        setattr(r, "scene", _FakeScene(n=2))
        setattr(r, "app", _FakeApp())
        r._static_entities["a"] = _FakeEntity(enabled=True)
        r._dynamic_entities["b"] = _FakeEntity(enabled=False)
        r._agent_entities["c"] = _FakeEntity(enabled=True)

        snap = r.get_debug_snapshot()
        self.assertEqual(snap["tracked_total"], 3)
        self.assertEqual(snap["visible_total"], 2)

    def test_camera_validity_flag(self):
        r = Ursina3DRenderer(24, 24)
        setattr(r, "camera", _FakeCamera(position=(1, 2, 3), rotation=(10, 20, 0), fov=60))
        setattr(r, "scene", _FakeScene(n=1))
        setattr(r, "app", _FakeApp())

        snap = r.get_debug_snapshot()
        self.assertTrue(snap["camera_valid"])
        self.assertEqual(tuple(snap["camera_pos"]), (1, 2, 3))
        self.assertEqual(tuple(snap["camera_rot"]), (10, 20, 0))

    def test_update_loop_presence_flag(self):
        r = Ursina3DRenderer(24, 24)
        setattr(r, "camera", _FakeCamera())
        setattr(r, "scene", _FakeScene(n=1))

        setattr(r, "app", _FakeApp())
        self.assertTrue(r.get_debug_snapshot()["app_has_step"])

        setattr(r, "app", _NoStepApp())
        self.assertFalse(r.get_debug_snapshot()["app_has_step"])

    def test_draw_executes_update_loop(self):
        r = Ursina3DRenderer(8, 8)
        setattr(r, "camera", _FakeCamera())
        setattr(r, "scene", _FakeScene(n=1))
        app = _FakeApp()
        setattr(r, "app", app)

        r._rebuild_static = lambda grid, h, w: r._static_entities.update({"/floor": _FakeEntity(True)})
        r._update_dynamic = lambda grid, h, w: None
        r._update_agents = lambda state, h, w: None
        r._ensure_fail_safe_visual = lambda force=False: None

        state = {
            "grid": np.zeros((8, 8), dtype=np.int32),
            "agents": [],
        }
        ok = r.draw(state, fps=0)
        self.assertTrue(ok)
        self.assertEqual(app.step_calls, 1)
        self.assertEqual(r._frame_idx, 1)


if __name__ == "__main__":
    unittest.main()
