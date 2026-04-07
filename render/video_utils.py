from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from env.world import WALL, EMPTY, DOOR, HIDE_SPOT, LIGHT_SW, FOOD, FAKE_FOOD, BARRICADE, HEAVY_OBJ, SCENT
from env.agent import Team


_TILE_COLORS = {
    WALL: (55, 55, 70),
    EMPTY: (38, 38, 52),
    DOOR: (120, 90, 50),
    HIDE_SPOT: (40, 70, 60),
    LIGHT_SW: (220, 200, 80),
    FOOD: (80, 200, 80),
    FAKE_FOOD: (180, 200, 80),
    BARRICADE: (80, 50, 30),
    HEAVY_OBJ: (140, 80, 40),
    SCENT: (38, 38, 52),
}

C_BG = (20, 20, 30)
C_FLOOR_DARK = (18, 18, 32)
C_HIDER = (50, 220, 120)
C_SEEKER = (220, 60, 60)
C_DEAD = (200, 50, 50)


def _as_np_2d(arr: Any, dtype) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr.astype(dtype, copy=False)
    return np.asarray(arr, dtype=dtype)


def _room_light_lookup(state: Dict[str, Any]) -> Dict[int, bool]:
    if "room_lights" in state:
        raw = state["room_lights"]
        if isinstance(raw, dict):
            return {int(k): bool(v) for k, v in raw.items()}
        return {int(k): bool(v) for k, v in raw}

    rooms = state.get("rooms", [])
    lookup: Dict[int, bool] = {}
    for room in rooms:
        if isinstance(room, dict):
            lookup[int(room["room_id"])] = bool(room.get("light_on", True))
        else:
            lookup[int(room.room_id)] = bool(room.light_on)
    return lookup


def _tile_to_room_lookup(state: Dict[str, Any]) -> Dict[Tuple[int, int], int]:
    raw = state.get("tile_to_room", {})
    if isinstance(raw, dict):
        return {(int(r), int(c)): int(v) for (r, c), v in raw.items()}

    out: Dict[Tuple[int, int], int] = {}
    for item in raw:
        if len(item) != 3:
            continue
        r, c, rid = item
        out[(int(r), int(c))] = int(rid)
    return out


def render_state_to_frame(state: Dict[str, Any], tile_px: int = 10, hud_px: int = 24) -> np.ndarray:
    grid = _as_np_2d(state["grid"], np.int32)
    scent_map = _as_np_2d(state["scent_map"], np.float32)
    h, w = grid.shape

    frame = np.zeros((h * tile_px + hud_px, w * tile_px, 3), dtype=np.uint8)
    frame[:, :] = np.array(C_BG, dtype=np.uint8)

    lit_rooms = _room_light_lookup(state)
    tile_to_room = _tile_to_room_lookup(state)

    for r in range(h):
        for c in range(w):
            tile = int(grid[r, c])
            color = _TILE_COLORS.get(tile, _TILE_COLORS[EMPTY])
            rid = tile_to_room.get((r, c))
            if rid is not None and not lit_rooms.get(rid, True):
                if tile in (HIDE_SPOT, LIGHT_SW, FOOD, FAKE_FOOD, HEAVY_OBJ):
                    color = tuple(int(v * 0.45) for v in _TILE_COLORS.get(tile, _TILE_COLORS[EMPTY]))
                else:
                    color = C_FLOOR_DARK

            y0 = hud_px + r * tile_px
            x0 = c * tile_px
            frame[y0:y0 + tile_px, x0:x0 + tile_px] = color

            scent = float(scent_map[r, c])
            if scent > 0.05:
                alpha = min(0.6, scent * 0.6)
                overlay = np.array([255, 140, 0], dtype=np.float32)
                tile_region = frame[y0:y0 + tile_px, x0:x0 + tile_px].astype(np.float32)
                tile_region = (1 - alpha) * tile_region + alpha * overlay
                frame[y0:y0 + tile_px, x0:x0 + tile_px] = tile_region.astype(np.uint8)

    agents = state.get("agents", [])
    for a in agents:
        if isinstance(a, dict):
            row = int(a["row"])
            col = int(a["col"])
            team_val = int(a["team"])
            alive = bool(a.get("alive", True))
            stunned = bool(a.get("stunned", False))
        else:
            _, row, col, team_val, alive, _, stunned = a

        y0 = hud_px + row * tile_px
        x0 = col * tile_px
        if alive:
            color = C_HIDER if team_val == Team.HIDER.value else C_SEEKER
            if stunned:
                color = (220, 180, 50) if team_val == Team.HIDER.value else (180, 80, 200)
        else:
            color = C_DEAD

        pad = max(1, tile_px // 6)
        frame[y0 + pad:y0 + tile_px - pad, x0 + pad:x0 + tile_px - pad] = color

    return frame


def write_mp4(path: Path, frames: Iterable[np.ndarray], fps: int = 12) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import importlib
    imageio = importlib.import_module("imageio")

    def _pad_to_macro_block(frame: np.ndarray, macro_block_size: int = 16) -> np.ndarray:
        if macro_block_size <= 1:
            return frame
        h, w = frame.shape[:2]
        pad_h = (macro_block_size - (h % macro_block_size)) % macro_block_size
        pad_w = (macro_block_size - (w % macro_block_size)) % macro_block_size
        if pad_h == 0 and pad_w == 0:
            return frame

        if frame.ndim == 2:
            return np.pad(frame, ((0, pad_h), (0, pad_w)), mode="edge")
        if frame.ndim == 3:
            return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        return frame

    with imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        quality=7,
        macro_block_size=16,
    ) as writer:
        for frame in frames:
            writer.append_data(_pad_to_macro_block(frame, macro_block_size=16))
