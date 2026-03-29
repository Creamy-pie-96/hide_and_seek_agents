from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import time

import numpy as np

from env.world import WALL, DOOR, BARRICADE, HEAVY_OBJ, FOOD, FAKE_FOOD, LIGHT_SW, HIDE_SPOT
from env.agent import Team


@dataclass
class Renderer3DConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    tile_size: float = 1.0
    wall_height: float = 1.0


class Viser3DRenderer:
    """
    3D renderer wrapper for Viser.

    Notes:
    - This module only visualizes state; it has no game logic.
    - It accepts the same render-state dict produced by the environment.
    """

    def __init__(self, grid_h: int, grid_w: int, config: Optional[Renderer3DConfig] = None):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.config = config or Renderer3DConfig()
        self.server = None
        self._scene_handles: List[Any] = []
        self._last_time = None

    def init(self) -> None:
        try:
            import viser  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("viser is not installed. Run: pip install viser") from e

        self.server = viser.ViserServer(host=self.config.host, port=self.config.port)
        self._last_time = time.time()

    def _clear_scene(self) -> None:
        for h in self._scene_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._scene_handles.clear()

    def _add_box(self, name: str, position: Tuple[float, float, float], dims: Tuple[float, float, float], color: Tuple[int, int, int], opacity: float = 1.0) -> None:
        if self.server is None:
            return
        handle = self.server.scene.add_box(
            name,
            dimensions=dims,
            position=position,
            color=color,
            opacity=opacity,
        )
        self._scene_handles.append(handle)

    def _add_sphere(self, name: str, position: Tuple[float, float, float], radius: float, color: Tuple[int, int, int], opacity: float = 1.0) -> None:
        if self.server is None:
            return
        handle = self.server.scene.add_icosphere(
            name,
            radius=radius,
            position=position,
            color=color,
            opacity=opacity,
        )
        self._scene_handles.append(handle)

    def draw(self, state: Dict[str, Any], fps: int = 20) -> bool:
        if self.server is None:
            raise RuntimeError("3D renderer not initialized. Call init() first.")

        self._clear_scene()

        grid = np.asarray(state["grid"], dtype=np.int32)
        h, w = grid.shape
        ts = self.config.tile_size

        # floor
        self._add_box(
            "/floor",
            position=(w * ts * 0.5, h * ts * 0.5, -0.02),
            dims=(w * ts, h * ts, 0.04),
            color=(200, 200, 205),
            opacity=1.0,
        )

        # static tiles
        for r in range(h):
            for c in range(w):
                tile = int(grid[r, c])
                x = (c + 0.5) * ts
                y = (r + 0.5) * ts

                if tile == WALL:
                    self._add_box(f"/wall/{r}_{c}", (x, y, self.config.wall_height * 0.5), (ts, ts, self.config.wall_height), (150, 150, 160), 1.0)
                elif tile == DOOR:
                    self._add_box(f"/door/{r}_{c}", (x, y, 0.15), (ts * 0.9, ts * 0.9, 0.3), (180, 130, 80), 0.9)
                elif tile == BARRICADE:
                    self._add_box(f"/barr/{r}_{c}", (x, y, 0.2), (ts * 0.9, ts * 0.9, 0.4), (120, 70, 40), 1.0)
                elif tile == HEAVY_OBJ:
                    self._add_box(f"/heavy/{r}_{c}", (x, y, 0.25), (ts * 0.9, ts * 0.9, 0.5), (220, 180, 50), 1.0)
                elif tile in (FOOD, FAKE_FOOD, LIGHT_SW, HIDE_SPOT):
                    col = (90, 220, 90) if tile == FOOD else (190, 210, 100)
                    if tile == LIGHT_SW:
                        col = (240, 210, 90)
                    if tile == HIDE_SPOT:
                        col = (80, 140, 120)
                    self._add_box(f"/obj/{r}_{c}", (x, y, 0.08), (ts * 0.6, ts * 0.6, 0.16), col, 1.0)

        # agents
        for i, agent in enumerate(state.get("agents", [])):
            if isinstance(agent, dict):
                aid = int(agent["id"])
                row = int(agent["row"])
                col = int(agent["col"])
                team = int(agent["team"])
                alive = bool(agent.get("alive", True))
                stunned = bool(agent.get("stunned", False))
            else:
                aid, row, col, team, alive, _, stunned = agent

            x = (col + 0.5) * ts
            y = (row + 0.5) * ts
            z = 0.4
            if not alive:
                color = (200, 70, 70)
            elif team == Team.HIDER.value:
                color = (80, 220, 150) if not stunned else (220, 180, 80)
            else:
                color = (240, 100, 100) if not stunned else (180, 80, 200)

            self._add_sphere(f"/agent/{aid}_{i}", (x, y, z), radius=0.26 * ts, color=color, opacity=0.95)

        # basic frame pacing
        now = time.time()
        if self._last_time is not None and fps > 0:
            dt_target = 1.0 / float(fps)
            elapsed = now - self._last_time
            if elapsed < dt_target:
                time.sleep(dt_target - elapsed)
        self._last_time = time.time()
        return True

    def close(self) -> None:
        self._clear_scene()
