from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import time

import numpy as np

from env.world import WALL, DOOR, BARRICADE, HEAVY_OBJ, FOOD, FAKE_FOOD, LIGHT_SW, HIDE_SPOT
from env.agent import Team


@dataclass
class UrsinaRendererConfig:
    tile_size: float = 1.0
    wall_height: float = 1.0
    asset_dir: str = "assest"
    use_fbx_models: bool = False
    debug: bool = True
    agent_scale: float = 0.95
    show_failsafe_cube: bool = False


class Ursina3DRenderer:
    """
    Ursina-based 3D renderer using primitive blocks + optional FBX agent models.
    """

    def __init__(self, grid_h: int, grid_w: int, config: Optional[UrsinaRendererConfig] = None):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.config = config or UrsinaRendererConfig()

        self.app = None
        self.camera = None
        self.scene = None

        self._static_entities: Dict[str, Any] = {}
        self._dynamic_entities: Dict[str, Any] = {}
        self._agent_entities: Dict[str, Any] = {}
        self._last_grid_shape: Optional[Tuple[int, int]] = None
        self._last_time: Optional[float] = None
        self._debug_printed = False
        self._frame_idx = 0
        self._failsafe_entity_key = "/debug/failsafe_cube"

        self._hider_model = None
        self._seeker_model = None
        self._hider_model_path: Optional[Path] = None
        self._seeker_model_path: Optional[Path] = None

    def init(self) -> None:
        try:
            from ursina import Ursina, camera, window, color, AmbientLight, DirectionalLight, Vec3, scene
        except Exception as e:  # pragma: no cover
            raise ImportError("ursina is not installed. Run: pip install ursina") from e

        self.app = Ursina(borderless=False, fullscreen=False, title="Hide & Seek 3D")
        self.camera = camera
        self.scene = scene

        window.color = color.rgb(16, 20, 28)
        self._reset_camera_to_origin(reason="init")

        AmbientLight(parent=scene, color=color.rgba(90, 90, 90, 1))
        DirectionalLight(parent=scene, y=3, z=2, rotation=(35, -30, 0), color=color.rgba(130, 130, 130, 1))

        # Fail-safe: guarantee at least one visible object in front of camera.
        if self.config.show_failsafe_cube:
            self._ensure_fail_safe_visual(force=True)

        if self.config.use_fbx_models:
            self._try_load_models()
        self._last_time = time.time()

    def _try_load_models(self) -> None:
        try:
            from ursina import load_model
        except Exception:
            return

        asset_root = Path(self.config.asset_dir)
        hider = asset_root / "hider.fbx"
        seeker = asset_root / "seeker.fbx"

        try:
            if hider.exists():
                self._hider_model = load_model(str(hider))
                self._hider_model_path = hider
                if self.config.debug:
                    print(f"[ursina][asset] loaded hider model: {hider}")
        except Exception:
            self._hider_model = None
            self._hider_model_path = None

        try:
            if seeker.exists():
                self._seeker_model = load_model(str(seeker))
                self._seeker_model_path = seeker
                if self.config.debug:
                    print(f"[ursina][asset] loaded seeker model: {seeker}")
        except Exception:
            self._seeker_model = None
            self._seeker_model_path = None

    def _world_xz(self, row: int, col: int, h: int, w: int) -> Tuple[float, float]:
        ts = self.config.tile_size
        x = (col + 0.5 - (w / 2.0)) * ts
        z = (row + 0.5 - (h / 2.0)) * ts
        return x, z

    def _safe_enabled(self, ent: Any) -> bool:
        return bool(getattr(ent, "enabled", True))

    def _count_visible_entities(self, table: Dict[str, Any]) -> int:
        return sum(1 for ent in table.values() if self._safe_enabled(ent))

    def _reset_camera_to_origin(self, reason: str = "") -> None:
        if self.camera is None:
            return
        try:
            from ursina import Vec3
        except Exception:
            Vec3 = None

        span = float(max(self.grid_h, self.grid_w, 6))
        self.camera.position = (0, span * 0.95, -span * 1.05)
        if Vec3 is not None:
            self.camera.look_at(Vec3(0, 0, 0))
        if self.config.debug:
            suffix = f" ({reason})" if reason else ""
            print(f"[ursina][debug] camera reset to origin{suffix}")

    def _ensure_fail_safe_visual(self, force: bool = False) -> None:
        if self.app is None:
            return
        from ursina import Entity, color

        cube = self._static_entities.get(self._failsafe_entity_key)
        if cube is None or force:
            cube = Entity(
                model="cube",
                scale=(1.5, 1.5, 1.5),
                position=(0, 0, 0),
                color=color.rgb(255, 0, 0),
            )
            self._static_entities[self._failsafe_entity_key] = cube
            if self.config.debug:
                print("[ursina][debug] fail-safe red cube added at (0,0,0)")
        elif not self._safe_enabled(cube):
            cube.enabled = True
            cube.position = (0, 0, 0)

    def _entity_name(self, ent: Any) -> str:
        name = getattr(ent, "name", None)
        if name:
            return str(name)
        model = getattr(ent, "model", None)
        if model is None:
            return ent.__class__.__name__
        return f"{ent.__class__.__name__}:{model}"

    def _print_scene_inspection(self, full: bool = False) -> None:
        scene_entities = []
        if self.scene is not None:
            scene_entities = list(getattr(self.scene, "entities", []))

        scene_names = [self._entity_name(e) for e in scene_entities]
        scene_enabled = sum(1 for e in scene_entities if self._safe_enabled(e))
        tracked_names = list(self._static_entities.keys()) + list(self._dynamic_entities.keys()) + list(self._agent_entities.keys())
        tracked_enabled = (
            self._count_visible_entities(self._static_entities)
            + self._count_visible_entities(self._dynamic_entities)
            + self._count_visible_entities(self._agent_entities)
        )

        print(f"[ursina][scene] total_scene_entities={len(scene_entities)}")
        if full:
            print(f"[ursina][scene] scene_entities={scene_names}")
        else:
            preview = scene_names[:20]
            print(f"[ursina][scene] scene_entities_preview={preview} (+{max(0, len(scene_names)-len(preview))} more)")
        print(f"[ursina][scene] enabled_scene_entities={scene_enabled}")
        if full:
            print(f"[ursina][scene] tracked_entities={tracked_names}")
        else:
            tracked_preview = tracked_names[:20]
            print(f"[ursina][scene] tracked_entities_preview={tracked_preview} (+{max(0, len(tracked_names)-len(tracked_preview))} more)")
        print(f"[ursina][scene] enabled_tracked_entities={tracked_enabled}")

    def _print_camera_state(self) -> None:
        if self.camera is None:
            print("[ursina][camera] camera=None")
            return
        print(f"[ursina][camera] position={getattr(self.camera, 'position', None)}")
        print(f"[ursina][camera] forward={getattr(self.camera, 'forward', None)}")
        print(f"[ursina][camera] rotation={getattr(self.camera, 'rotation', None)}")

    def _get_debug_snapshot(self) -> Dict[str, Any]:
        static_visible = self._count_visible_entities(self._static_entities)
        dynamic_visible = self._count_visible_entities(self._dynamic_entities)
        agents_visible = self._count_visible_entities(self._agent_entities)
        cam_pos = getattr(self.camera, "position", None)
        cam_rot = getattr(self.camera, "rotation", None)

        scene_entities = None
        if self.scene is not None:
            scene_entities = len(getattr(self.scene, "entities", []))

        app_has_step = bool(callable(getattr(self.app, "step", None)))
        camera_valid = (self.camera is not None and cam_pos is not None and cam_rot is not None)

        return {
            "frame": self._frame_idx,
            "tracked_total": len(self._static_entities) + len(self._dynamic_entities) + len(self._agent_entities),
            "tracked_static": len(self._static_entities),
            "tracked_dynamic": len(self._dynamic_entities),
            "tracked_agents": len(self._agent_entities),
            "visible_static": static_visible,
            "visible_dynamic": dynamic_visible,
            "visible_agents": agents_visible,
            "visible_total": static_visible + dynamic_visible + agents_visible,
            "scene_entities": scene_entities,
            "camera_valid": camera_valid,
            "camera_pos": cam_pos,
            "camera_rot": cam_rot,
            "camera_fov": getattr(self.camera, "fov", None),
            "app_has_step": app_has_step,
        }

    def get_debug_snapshot(self) -> Dict[str, Any]:
        """Public helper for tests and runtime diagnostics."""
        return self._get_debug_snapshot()

    def _print_debug_snapshot(self, snap: Dict[str, Any]) -> None:
        print(
            "[ursina][debug] "
            f"frame={snap['frame']} tracked={snap['tracked_total']} "
            f"(s={snap['tracked_static']} d={snap['tracked_dynamic']} a={snap['tracked_agents']}) "
            f"visible={snap['visible_total']} "
            f"(s={snap['visible_static']} d={snap['visible_dynamic']} a={snap['visible_agents']}) "
            f"scene={snap['scene_entities']} camera_valid={snap['camera_valid']} "
            f"camera_pos={snap['camera_pos']} camera_rot={snap['camera_rot']} "
            f"fov={snap['camera_fov']} app_step={snap['app_has_step']}"
        )

    def _clear_entities(self, table: Dict[str, Any]) -> None:
        for ent in list(table.values()):
            try:
                ent.disable()
            except Exception:
                pass
        table.clear()

    def _rebuild_static(self, grid: np.ndarray, h: int, w: int) -> None:
        from ursina import Entity, color

        self._clear_entities(self._static_entities)
        ts = self.config.tile_size

        floor = Entity(
            model="cube",
            scale=(w * ts, 0.05, h * ts),
            position=(0, -0.03, 0),
            color=color.rgb(64, 74, 90),
        )
        self._static_entities["/floor"] = floor

        # Always-visible origin marker for camera/debug sanity.
        self._static_entities["/origin"] = Entity(
            model="cube",
            scale=(0.35, 0.35, 0.35),
            position=(0, 0.2, 0),
            color=color.rgb(255, 60, 60),
        )

        for r in range(h):
            for c in range(w):
                tile = int(grid[r, c])
                if tile != WALL:
                    continue
                x, z = self._world_xz(r, c, h, w)
                name = f"/wall/{r}_{c}"
                self._static_entities[name] = Entity(
                    model="cube",
                    scale=(ts, self.config.wall_height, ts),
                    position=(x, self.config.wall_height * 0.5, z),
                    color=color.rgb(122, 145, 186),
                )

    def _dynamic_spec(self, tile: int):
        from ursina import color
        if tile == DOOR:
            return (0.9, 0.9, 0.3), color.rgb(214, 168, 104), 0.15
        if tile == BARRICADE:
            return (0.9, 0.9, 0.45), color.rgb(160, 97, 73), 0.225
        if tile == HEAVY_OBJ:
            return (0.9, 0.9, 0.5), color.rgb(235, 192, 96), 0.25
        if tile == FOOD:
            return (0.55, 0.55, 0.2), color.rgb(97, 220, 132), 0.1
        if tile == FAKE_FOOD:
            return (0.55, 0.55, 0.2), color.rgb(224, 184, 102), 0.1
        if tile == LIGHT_SW:
            return (0.55, 0.55, 0.2), color.rgb(250, 234, 130), 0.1
        if tile == HIDE_SPOT:
            return (0.7, 0.7, 0.2), color.rgb(89, 171, 140), 0.1
        return None

    def _update_dynamic(self, grid: np.ndarray, h: int, w: int) -> None:
        from ursina import Entity

        wanted: Dict[str, int] = {}
        for r in range(h):
            for c in range(w):
                t = int(grid[r, c])
                if self._dynamic_spec(t) is not None:
                    wanted[f"/dyn/{r}_{c}"] = t

        for name in list(self._dynamic_entities.keys()):
            if name not in wanted:
                try:
                    self._dynamic_entities[name].disable()
                except Exception:
                    pass
                self._dynamic_entities.pop(name, None)

        for r in range(h):
            for c in range(w):
                name = f"/dyn/{r}_{c}"
                t = int(grid[r, c])
                spec = self._dynamic_spec(t)
                ent = self._dynamic_entities.get(name)
                if spec is None:
                    continue

                scale_xyz, col, y = spec
                x, z = self._world_xz(r, c, h, w)

                if ent is None:
                    self._dynamic_entities[name] = Entity(
                        model="cube",
                        scale=scale_xyz,
                        position=(x, y, z),
                        color=col,
                    )
                else:
                    ent.scale = scale_xyz
                    ent.position = (x, y, z)
                    ent.color = col

    def _make_agent_entity(self, name: str, team: int):
        from ursina import Entity, color, load_model

        model = None
        if team == Team.HIDER.value and self._hider_model is not None:
            model = self._hider_model
        if team == Team.SEEKER.value and self._seeker_model is not None:
            model = self._seeker_model

        if model is not None:
            model_path: Optional[Path] = self._hider_model_path if team == Team.HIDER.value else self._seeker_model_path
            if model_path is not None:
                try:
                    # Load a fresh copy per agent to avoid NodePath re-parent side effects.
                    model_instance = load_model(str(model_path), use_deepcopy=True)
                except Exception:
                    model_instance = model
            else:
                model_instance = model

            ent = Entity(model=model_instance, scale=self.config.agent_scale)
            ent.rotation_x = 0
            ent.rotation_y = 180
            ent._is_fbx = True
            if self.config.debug:
                team_name = "hider" if team == Team.HIDER.value else "seeker"
                print(f"[ursina][asset] spawned {team_name} asset entity: {name}")
            return ent

        fallback = color.rgb(80, 220, 150) if team == Team.HIDER.value else color.rgb(240, 100, 100)
        ent = Entity(model="cube", scale=(0.5, 0.5, 0.8), color=fallback)
        ent._is_fbx = False
        return ent

    def _update_agents(self, state: Dict[str, Any], h: int, w: int) -> None:
        from ursina import color

        seen = set()
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

            name = f"/agent/{aid}_{i}"
            seen.add(name)
            ent = self._agent_entities.get(name)
            if ent is None:
                ent = self._make_agent_entity(name, team)
                self._agent_entities[name] = ent

            x, z = self._world_xz(row, col, h, w)
            is_fbx = bool(getattr(ent, "_is_fbx", False))
            y = self.config.agent_scale * 1.1 if is_fbx else 0.45
            ent.position = (x, y, z)

            if is_fbx:
                if not alive:
                    ent.color = color.rgba(130, 130, 130, 1)
                elif stunned:
                    ent.color = color.rgba(255, 210, 80, 1)
                else:
                    ent.color = color.rgba(255, 255, 255, 1)
            else:
                if not alive:
                    ent.color = color.rgb(200, 70, 70)
                elif team == Team.HIDER.value:
                    ent.color = color.rgb(220, 180, 80) if stunned else color.rgb(80, 220, 150)
                else:
                    ent.color = color.rgb(180, 80, 200) if stunned else color.rgb(240, 100, 100)

        for name in list(self._agent_entities.keys()):
            if name not in seen:
                try:
                    self._agent_entities[name].disable()
                except Exception:
                    pass
                self._agent_entities.pop(name, None)

    def draw(self, state: Dict[str, Any], fps: int = 20) -> bool:
        if self.app is None:
            raise RuntimeError("Ursina renderer not initialized. Call init() first.")

        grid = np.asarray(state["grid"], dtype=np.int32)
        h, w = grid.shape

        if self._last_grid_shape != (h, w):
            self._clear_entities(self._dynamic_entities)
            self._clear_entities(self._agent_entities)
            self._rebuild_static(grid, h, w)
            self._last_grid_shape = (h, w)

        self._update_dynamic(grid, h, w)
        self._update_agents(state, h, w)

        if self.config.debug and (self._frame_idx == 0 or self._frame_idx % 120 == 0):
            self._print_scene_inspection(full=(self._frame_idx == 0))
            self._print_camera_state()

        snap = self._get_debug_snapshot()
        if self.config.debug and (not self._debug_printed or self._frame_idx < 3 or self._frame_idx % 120 == 0):
            self._print_debug_snapshot(snap)
            self._debug_printed = True

        if not snap["camera_valid"]:
            self._reset_camera_to_origin(reason="invalid-camera")
        if snap["visible_total"] <= 0:
            if self.config.debug:
                print("[ursina][warn] no visible entities detected; forcing camera + fail-safe cube")
            self._ensure_fail_safe_visual(force=True)
            self._reset_camera_to_origin(reason="no-visible-entities")
        if not snap["app_has_step"]:
            if self.config.debug:
                print("[ursina][error] app.step() missing; update loop invalid")
            return False

        if self.config.debug and (self._frame_idx < 3 or self._frame_idx % 120 == 0):
            print(f"[ursina][update] frame={self._frame_idx} draw loop executing")

        try:
            self.app.step()
            self._frame_idx += 1
        except Exception:
            return False

        now = time.time()
        if self._last_time is not None and fps > 0:
            target = 1.0 / float(fps)
            elapsed = now - self._last_time
            if elapsed < target:
                time.sleep(target - elapsed)
        self._last_time = time.time()
        return True

    def close(self) -> None:
        self._clear_entities(self._agent_entities)
        self._clear_entities(self._dynamic_entities)
        self._clear_entities(self._static_entities)
        if self.app is not None:
            try:
                self.app.userExit()
            except Exception:
                pass
            self.app = None
