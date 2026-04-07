from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np

from sim2.entities import Tile


def generate_layout(width: int, height: int, seed: int) -> np.ndarray:
    """Deterministic, readable Pac-Man-like layout."""
    rng = random.Random(seed)
    grid = np.full((height, width), int(Tile.EMPTY), dtype=np.int32)

    # Border walls.
    grid[0, :] = int(Tile.WALL)
    grid[-1, :] = int(Tile.WALL)
    grid[:, 0] = int(Tile.WALL)
    grid[:, -1] = int(Tile.WALL)

    # Vertical corridors every 4 columns, with deterministic door holes.
    for c in range(3, width - 1, 4):
        grid[1:-1, c] = int(Tile.WALL)
        hole_rows = [height // 4, height // 2, (3 * height) // 4]
        for hr in hole_rows:
            hr = max(1, min(height - 2, hr + rng.randint(-1, 1)))
            grid[hr, c] = int(Tile.DOOR)

    # Horizontal corridors every 4 rows, with deterministic door holes.
    for r in range(3, height - 1, 4):
        grid[r, 1:-1] = int(Tile.WALL)
        hole_cols = [width // 4, width // 2, (3 * width) // 4]
        for hc in hole_cols:
            hc = max(1, min(width - 2, hc + rng.randint(-1, 1)))
            grid[r, hc] = int(Tile.DOOR)

    # Place a few hide spots and food in empty tiles.
    empties = list(zip(*np.where(grid == int(Tile.EMPTY))))
    rng.shuffle(empties)
    for pos in empties[: max(2, (width * height) // 80)]:
        grid[pos] = int(Tile.HIDE_SPOT)
    for pos in empties[max(2, (width * height) // 80): max(6, (width * height) // 50)]:
        if grid[pos] == int(Tile.EMPTY):
            grid[pos] = int(Tile.FOOD)

    return grid


def default_spawns(grid: np.ndarray) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    h, w = grid.shape
    empties = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1) if grid[r, c] != int(Tile.WALL)]
    empties.sort()

    hider_spawns = [empties[0], empties[1], empties[2]]
    seeker_spawns = [empties[-1], empties[-2], empties[-3]]
    return hider_spawns, seeker_spawns
