from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import time

import numpy as np

from sim2.entities import Team, Tile


@dataclass
class PrimitiveUrsinaConfig:
    tile_size: float = 1.0
    wall_height: float = 1.0
    fps: int = 20
    debug: bool = True


class PrimitiveUrsinaRenderer:
    """Minimal model-free Ursina renderer for sim2 snapshots."""

    def __init__(self, grid_h: int, grid_w: int, config: Optional[PrimitiveUrsinaConfig] = None):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.config = config or PrimitiveUrsinaConfig()

        self.app = None
        self.camera = None
        self.scene = None

        self._static: Dict[str, Any] = {}
        self._agents: Dict[int, Any] = {}
        self._grid_shape: Optional[Tuple[int, int]] = None
        self._last_time = None
        self._frame = 0

    def init(self) -> None:
        try:
            from ursina import Ursina, AmbientLight, DirectionalLight, camera, color, scene, window, Vec3
        except Exception as e:
            raise ImportError("ursina is not installed. Run: pip install ursina") from e

        self.app = Ursina(borderless=False, fullscreen=False, title="Hide & Seek Sim2")
        self.camera = camera
        self.scene = scene

        window.color = color.rgb(26, 30, 40)
        span = float(max(self.grid_h, self.grid_w, 8))
        camera.position = (0, span * 1.05, -span * 1.1)
        camera.fov = 70
        camera.look_at(Vec3(0, 0, 0))

        AmbientLight(parent=scene, color=color.rgba(150, 150, 150, 1))
        DirectionalLight(parent=scene, y=3, z=2, rotation=(35, -30, 0), color=color.rgba(200, 200, 200, 1))

        self._last_time = time.time()

    def _world_xz(self, row: int, col: int, h: int, w: int) -> Tuple[float, float]:
        ts = self.config.tile_size
        x = (col + 0.5 - (w / 2.0)) * ts
        z = (row + 0.5 - (h / 2.0)) * ts
        return x, z

    def _clear_table(self, table: Dict[Any, Any]) -> None:
        for ent in list(table.values()):
            try:
                ent.disable()
            except Exception:
                pass
        table.clear()

    def _rebuild_static(self, grid: np.ndarray, h: int, w: int) -> None:
        from ursina import Entity, color

        self._clear_table(self._static)
        ts = self.config.tile_size

        self._static["floor"] = Entity(
            model="cube",
            scale=(w * ts, 0.05, h * ts),
            position=(0, -0.03, 0),
            color=color.rgb(90, 104, 128),
        )

        # Guaranteed visible marker for camera/render sanity.
        self._static["origin_marker"] = Entity(
            model="cube",
            scale=(0.6, 0.6, 0.6),
            position=(0, 0.35, 0),
            color=color.rgb(255, 60, 60),
        )

        for r in range(h):
            for c in range(w):
                tile = int(grid[r, c])
                x, z = self._world_xz(r, c, h, w)
                if tile == int(Tile.WALL):
                    self._static[f"wall/{r}/{c}"] = Entity(
                        model="cube",
                        scale=(ts, self.config.wall_height, ts),
                        position=(x, self.config.wall_height * 0.5, z),
                        color=color.rgb(122, 145, 186),
                    )
                elif tile == int(Tile.DOOR):
                    self._static[f"door/{r}/{c}"] = Entity(
                        model="cube",
                        scale=(0.8 * ts, 0.35, 0.8 * ts),
                        position=(x, 0.18, z),
                        color=color.rgb(180, 130, 70),
                    )
                elif tile == int(Tile.HIDE_SPOT):
                    self._static[f"hide/{r}/{c}"] = Entity(
                        model="cube",
                        scale=(0.6 * ts, 0.25, 0.6 * ts),
                        position=(x, 0.12, z),
                        color=color.rgb(70, 140, 115),
                    )
                elif tile == int(Tile.FOOD):
                    self._static[f"food/{r}/{c}"] = Entity(
                        model="sphere",
                        scale=(0.22, 0.22, 0.22),
                        position=(x, 0.16, z),
                        color=color.rgb(96, 220, 96),
                    )

    def _update_agents(self, state: Dict[str, Any], h: int, w: int) -> None:
        from ursina import Entity, color

        seen = set()
        for a in state.get("agents", []):
            aid = int(a["id"])
            team = int(a["team"])
            alive = bool(a.get("alive", True))
            row = int(a["row"])
            col = int(a["col"])

            seen.add(aid)
            ent = self._agents.get(aid)
            if ent is None:
                base = color.rgb(70, 220, 150) if team == int(Team.HIDER) else color.rgb(240, 110, 110)
                ent = Entity(model="sphere", scale=(0.46, 0.46, 0.46), color=base)
                self._agents[aid] = ent

            x, z = self._world_xz(row, col, h, w)
            ent.position = (x, 0.32, z)
            if not alive:
                ent.color = color.rgb(120, 120, 120)

        for aid in list(self._agents.keys()):
            if aid not in seen:
                try:
                    self._agents[aid].disable()
                except Exception:
                    pass
                self._agents.pop(aid, None)

    def draw(self, state: Dict[str, Any], fps: Optional[int] = None) -> bool:
        if self.app is None:
            raise RuntimeError("Primitive renderer not initialized. Call init() first.")

        grid = np.asarray(state["grid"], dtype=np.int32)
        h, w = grid.shape

        if self._grid_shape != (h, w):
            self._rebuild_static(grid, h, w)
            self._grid_shape = (h, w)

        self._update_agents(state, h, w)

        if self.config.debug and (self._frame < 3 or self._frame % 120 == 0):
            print(
                f"[sim2][ursina] frame={self._frame} static={len(self._static)} agents={len(self._agents)} "
                f"camera_pos={getattr(self.camera, 'position', None)}"
            )

        try:
            self.app.step()
            self._frame += 1
        except Exception:
            return False

        now = time.time()
        target_fps = self.config.fps if fps is None else fps
        if self._last_time is not None and target_fps > 0:
            target = 1.0 / float(target_fps)
            elapsed = now - self._last_time
            if elapsed < target:
                time.sleep(target - elapsed)
        self._last_time = time.time()
        return True

    def close(self) -> None:
        self._clear_table(self._agents)
        self._clear_table(self._static)
        if self.app is not None:
            try:
                self.app.userExit()
            except Exception:
                pass
            self.app = None
