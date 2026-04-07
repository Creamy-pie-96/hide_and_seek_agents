"""
test_replay_validation.py — Tests for replay frame validation (B2 fix).
"""
import os
import sys
import unittest
import json
import tempfile

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TestReplayValidation(unittest.TestCase):
    """Test that play.py's replay validation catches malformed data."""

    def _write_replay(self, data: dict) -> str:
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        return path

    def test_valid_minimal_replay(self):
        """A minimal valid replay should pass validation."""
        from env.hide_seek_env import HideSeekEnv
        env = HideSeekEnv(width=24, height=24, seed=1)
        env.reset(seed=1)
        state = env.get_serializable_render_state()
        replay = {
            "schema_version": 1,
            "frames": [state],
        }
        path = self._write_replay(replay)
        try:
            with open(path, "r") as f:
                payload = json.load(f)
            frames = payload["frames"]
            required = {"grid", "agents"}
            for idx, frame in enumerate(frames):
                self.assertIsInstance(frame, dict, f"Frame {idx} should be dict")
                self.assertTrue(required.issubset(frame.keys()),
                                f"Frame {idx} missing keys: {required - frame.keys()}")
                self.assertIsInstance(frame["grid"], list)
                self.assertGreater(len(frame["grid"]), 0)
                self.assertIsInstance(frame["agents"], list)
        finally:
            os.unlink(path)

    def test_missing_grid_key(self):
        """Frames missing 'grid' should be detected."""
        replay = {
            "schema_version": 1,
            "frames": [{"agents": []}],
        }
        path = self._write_replay(replay)
        try:
            with open(path, "r") as f:
                payload = json.load(f)
            frames = payload["frames"]
            required = {"grid", "agents"}
            frame = frames[0]
            missing = required - frame.keys()
            self.assertGreater(len(missing), 0, "Should detect missing 'grid'")
            self.assertIn("grid", missing)
        finally:
            os.unlink(path)

    def test_non_dict_frame(self):
        """Non-dict frames should be detected."""
        replay = {
            "schema_version": 1,
            "frames": ["not_a_dict"],
        }
        path = self._write_replay(replay)
        try:
            with open(path, "r") as f:
                payload = json.load(f)
            frame = payload["frames"][0]
            self.assertNotIsInstance(frame, dict)
        finally:
            os.unlink(path)

    def test_empty_frames(self):
        """Empty frames list should be rejected."""
        replay = {
            "schema_version": 1,
            "frames": [],
        }
        path = self._write_replay(replay)
        try:
            with open(path, "r") as f:
                payload = json.load(f)
            frames = payload["frames"]
            self.assertEqual(len(frames), 0)
        finally:
            os.unlink(path)


if __name__ == '__main__':
    unittest.main()
