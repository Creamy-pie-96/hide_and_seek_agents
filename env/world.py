"""
world.py — Procedural map generator using Binary Space Partitioning (BSP).

Each episode generates a fresh layout:
  - Rooms of varying sizes connected by corridors
  - Doors between rooms (can be barricaded)
  - Light switches per room (on by default)
  - Hiding spots (alcoves, crates)
  - Food spawns scattered around
  - Heavy barricade objects (require 2 agents to push)

Tile legend (int):
  0  EMPTY       walkable floor
  1  WALL        solid, blocks movement and vision
  2  DOOR        walkable, can be barricaded
  3  HIDE_SPOT   walkable, hiders are invisible here unless adjacen
  4  LIGHT_SW    walkable, toggle to control room lighting
  5  FOOD        walkable, consumed on contact
  6  FAKE_FOOD   walkable, placed by hiders as decoy
  7  BARRICADE   solid after placed, blocks movement
  8  HEAVY_OBJ   requires 2 adjacent agents to push (cooperative)
  9  SCENT       floor marker, fades over time
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

# ── Tile constants ──────────────────────────────────────────────────────────
EMPTY      = 0
WALL       = 1
DOOR       = 2
HIDE_SPOT  = 3
LIGHT_SW   = 4
FOOD       = 5
FAKE_FOOD  = 6
BARRICADE  = 7
HEAVY_OBJ  = 8
SCENT      = 9

WALKABLE = {EMPTY, DOOR, HIDE_SPOT, LIGHT_SW, FOOD, FAKE_FOOD, SCENT}
SOLID    = {WALL, BARRICADE}


@dataclass
class Room:
    x: int          # top-left col
    y: int          # top-left row
    w: int          # width  (cols)
    h: int          # height (rows)
    light_on: bool  = True
    light_switch_pos: Optional[Tuple[int,int]] = None
    door_positions: List[Tuple[int,int]] = field(default_factory=list)
    hide_spots:    List[Tuple[int,int]] = field(default_factory=list)
    food_positions:List[Tuple[int,int]] = field(default_factory=list)
    room_id: int = 0

    @property
    def center(self) -> Tuple[int,int]:
        return (self.y + self.h // 2, self.x + self.w // 2)

    def inner_tiles(self) -> List[Tuple[int,int]]:
        """All floor tiles strictly inside walls."""
        tiles = []
        for r in range(self.y + 1, self.y + self.h - 1):
            for c in range(self.x + 1, self.x + self.w - 1):
                tiles.append((r, c))
        return tiles

    def contains(self, row: int, col: int) -> bool:
        return self.x < col < self.x + self.w - 1 and self.y < row < self.y + self.h - 1


class BSPNode:
    """Node in the BSP tree. Leaf nodes become rooms."""
    MIN_SIZE = 8

    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.left:  Optional['BSPNode'] = None
        self.right: Optional['BSPNode'] = None
        self.room:  Optional[Room]      = None

    def split(self, rng: random.Random) -> bool:
        if self.left or self.right:
            return False  # already split

        # Decide split direction: prefer splitting the longer axis
        split_h = rng.random() > 0.5
        if self.w > self.h and self.w / self.h >= 1.25:
            split_h = False
        elif self.h > self.w and self.h / self.w >= 1.25:
            split_h = True

        max_size = (self.h if split_h else self.w) - self.MIN_SIZE
        if max_size <= self.MIN_SIZE:
            return False  # too small to split

        split_at = rng.randint(self.MIN_SIZE, max_size)

        if split_h:
            self.left  = BSPNode(self.x, self.y,              self.w, split_at)
            self.right = BSPNode(self.x, self.y + split_at,   self.w, self.h - split_at)
        else:
            self.left  = BSPNode(self.x,              self.y, split_at,          self.h)
            self.right = BSPNode(self.x + split_at,   self.y, self.w - split_at, self.h)
        return True

    def get_leaves(self) -> List['BSPNode']:
        if not self.left and not self.right:
            return [self]
        leaves = []
        if self.left:  leaves.extend(self.left.get_leaves())
        if self.right: leaves.extend(self.right.get_leaves())
        return leaves

    def get_room(self) -> Optional[Room]:
        """Return the room carved in this node or recursively in children."""
        if self.room:
            return self.room
        left_room  = self.left.get_room()  if self.left  else None
        right_room = self.right.get_room() if self.right else None
        if not left_room:  return right_room
        if not right_room: return left_room
        # Return the one closer to center of node for corridor routing
        return left_room


class WorldGenerator:
    """
    Generates a complete game map each episode.

    Parameters
    ----------
    width, height  : map dimensions in tiles
    seed           : optional RNG seed (None = random each call)
    n_food         : food items to scatter
    n_heavy_obj    : heavy barricade objects needing 2-agent push
    """

    def __init__(self, width: int = 48, height: int = 48,
                 n_food: int = 12, n_heavy_obj: int = 4):
        self.width     = width
        self.height    = height
        self.n_food    = n_food
        self.n_heavy_obj = n_heavy_obj

    def generate(self, seed: Optional[int] = None) -> 'World':
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)

        grid = np.ones((self.height, self.width), dtype=np.int32)  # all walls

        # ── 1. BSP split ────────────────────────────────────────────────────
        root = BSPNode(0, 0, self.width, self.height)
        nodes = [root]
        for _ in range(6):  # 6 splits → up to 64 potential rooms, realistically 8-12
            new_nodes = []
            for node in nodes:
                if node.split(rng):
                    new_nodes.extend([node.left, node.right])
                else:
                    new_nodes.append(node)
            nodes = new_nodes

        # ── 2. Carve rooms in leaves ─────────────────────────────────────────
        rooms: List[Room] = []
        for idx, leaf in enumerate(root.get_leaves()):
            margin = 1
            rw = rng.randint(5, max(5, leaf.w - margin * 2))
            rh = rng.randint(5, max(5, leaf.h - margin * 2))
            rx = leaf.x + rng.randint(margin, max(margin, leaf.w - rw - margin))
            ry = leaf.y + rng.randint(margin, max(margin, leaf.h - rh - margin))

            # Clamp to grid boundaries
            rx = max(1, min(rx, self.width  - rw - 1))
            ry = max(1, min(ry, self.height - rh - 1))

            room = Room(x=rx, y=ry, w=rw, h=rh, room_id=idx)
            leaf.room = room
            rooms.append(room)

            # Carve floor
            for r in range(ry, ry + rh):
                for c in range(rx, rx + rw):
                    grid[r, c] = EMPTY

        # ── 3. Carve corridors ───────────────────────────────────────────────
        door_positions: List[Tuple[int,int]] = []
        self._connect_rooms(root, grid, rng, door_positions)

        # ── 4. Place light switches ──────────────────────────────────────────
        for room in rooms:
            inner = room.inner_tiles()
            if inner:
                sw_pos = rng.choice(inner)
                grid[sw_pos[0], sw_pos[1]] = LIGHT_SW
                room.light_switch_pos = sw_pos

        # ── 5. Place hide spots (alcoves carved into walls) ──────────────────
        for room in rooms:
            n_spots = rng.randint(1, 3)
            candidates = self._wall_adjacent_tiles(room, grid)
            chosen = rng.sample(candidates, min(n_spots, len(candidates)))
            for pos in chosen:
                grid[pos[0], pos[1]] = HIDE_SPOT
                room.hide_spots.append(pos)

        # ── 6. Scatter food ──────────────────────────────────────────────────
        floor_tiles = list(zip(*np.where(grid == EMPTY)))
        rng.shuffle(floor_tiles)
        food_tiles = floor_tiles[:self.n_food]
        for r, c in food_tiles:
            grid[r, c] = FOOD
            # Track which room this food belongs to
            for room in rooms:
                if room.contains(r, c):
                    room.food_positions.append((r, c))
                    break

        # ── 7. Place heavy objects (need 2 agents to push) ───────────────────
        remaining_floor = floor_tiles[self.n_food:]
        heavy_tiles = remaining_floor[:self.n_heavy_obj]
        heavy_positions = []
        for r, c in heavy_tiles:
            grid[r, c] = HEAVY_OBJ
            heavy_positions.append((r, c))

        # ── 8. Place doors at corridor junctions ─────────────────────────────
        for pos in door_positions:
            r, c = pos
            if grid[r, c] == EMPTY:
                grid[r, c] = DOOR
            for room in rooms:
                if room.contains(r, c):
                    room.door_positions.append(pos)

        # ── 9. Build room lookup: tile → room_id ─────────────────────────────
        tile_to_room: Dict[Tuple[int,int], int] = {}
        for room in rooms:
            for r, c in room.inner_tiles():
                tile_to_room[(r, c)] = room.room_id

        return World(
            grid=grid,
            rooms=rooms,
            door_positions=door_positions,
            heavy_positions=heavy_positions,
            tile_to_room=tile_to_room,
            width=self.width,
            height=self.height,
        )

    # ── Private helpers ──────────────────────────────────────────────────────

    def _connect_rooms(self, node: BSPNode, grid: np.ndarray,
                       rng: random.Random,
                       door_positions: List[Tuple[int,int]]) -> None:
        """Recursively connect sibling rooms with L-shaped corridors."""
        if not node.left or not node.right:
            return
        self._connect_rooms(node.left,  grid, rng, door_positions)
        self._connect_rooms(node.right, grid, rng, door_positions)

        left_room  = node.left.get_room()
        right_room = node.right.get_room()
        if not left_room or not right_room:
            return

        r1, c1 = left_room.center
        r2, c2 = right_room.center

        # L-shaped corridor: horizontal then vertical (or v then h randomly)
        door_candidates = []
        if rng.random() > 0.5:
            door_candidates = self._carve_h_then_v(grid, r1, c1, r2, c2)
        else:
            door_candidates = self._carve_v_then_h(grid, r1, c1, r2, c2)

        # Place a door roughly in the middle of the corridor
        if door_candidates:
            mid = door_candidates[len(door_candidates) // 2]
            door_positions.append(mid)

    def _carve_h_then_v(self, grid, r1, c1, r2, c2) -> List[Tuple[int,int]]:
        corridor = []
        for c in range(min(c1, c2), max(c1, c2) + 1):
            if grid[r1, c] == WALL:
                grid[r1, c] = EMPTY
                corridor.append((r1, c))
        for r in range(min(r1, r2), max(r1, r2) + 1):
            if grid[r, c2] == WALL:
                grid[r, c2] = EMPTY
                corridor.append((r, c2))
        return corridor

    def _carve_v_then_h(self, grid, r1, c1, r2, c2) -> List[Tuple[int,int]]:
        corridor = []
        for r in range(min(r1, r2), max(r1, r2) + 1):
            if grid[r, c1] == WALL:
                grid[r, c1] = EMPTY
                corridor.append((r, c1))
        for c in range(min(c1, c2), max(c1, c2) + 1):
            if grid[r2, c] == WALL:
                grid[r2, c] = EMPTY
                corridor.append((r2, c))
        return corridor

    def _wall_adjacent_tiles(self, room: Room,
                             grid: np.ndarray) -> List[Tuple[int,int]]:
        """Floor tiles along the inner edge of a room — good hide spots."""
        candidates = []
        for r in range(room.y + 1, room.y + room.h - 1):
            for c in range(room.x + 1, room.x + room.w - 1):
                if grid[r, c] == EMPTY:
                    # Check if at least one neighbour is a wall
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        if grid[r+dr, c+dc] == WALL:
                            candidates.append((r, c))
                            break
        return candidates


class World:
    """
    The live game world. Mutated during an episode (lights toggle, food eaten,
    barricades placed, scent fades).
    """

    def __init__(self, grid: np.ndarray, rooms: List[Room],
                 door_positions: List[Tuple[int,int]],
                 heavy_positions: List[Tuple[int,int]],
                 tile_to_room: Dict[Tuple[int,int], int],
                 width: int, height: int):
        self.grid          = grid.copy()
        self._base_grid    = grid.copy()   # for reset
        self.rooms         = rooms
        self.door_positions = door_positions
        self.heavy_positions = list(heavy_positions)
        self.tile_to_room  = tile_to_room
        self.width         = width
        self.height        = height

        # Dynamic state
        self.barricaded_doors: set = set()   # (r,c) doors that are blocked
        self.scent_map  = np.zeros((height, width), dtype=np.float32)
        self.scent_ttl  = np.zeros((height, width), dtype=np.int32)
        self.fake_food  : List[Tuple[int,int]] = []

    # ── Tile queries ─────────────────────────────────────────────────────────

    def is_walkable(self, r: int, c: int) -> bool:
        if not (0 <= r < self.height and 0 <= c < self.width):
            return False
        t = self.grid[r, c]
        if t in SOLID:
            return False
        if (r, c) in self.barricaded_doors:
            return False
        return True

    def is_lit(self, r: int, c: int) -> bool:
        """Is a tile in a lit room? Corridor tiles always lit."""
        rid = self.tile_to_room.get((r, c))
        if rid is None:
            return True
        return self.rooms[rid].light_on

    def get_room(self, r: int, c: int) -> Optional[Room]:
        rid = self.tile_to_room.get((r, c))
        return self.rooms[rid] if rid is not None else None

    # ── Actions ──────────────────────────────────────────────────────────────

    def toggle_light(self, r: int, c: int) -> bool:
        """Toggle the light in whichever room contains (r,c)."""
        room = self.get_room(r, c)
        if room is None:
            return False
        room.light_on = not room.light_on
        return True

    def place_barricade(self, r: int, c: int) -> bool:
        if self.grid[r, c] == DOOR and (r, c) not in self.barricaded_doors:
            self.barricaded_doors.add((r, c))
            self.grid[r, c] = BARRICADE
            return True
        return False

    def remove_barricade(self, r: int, c: int) -> bool:
        if (r, c) in self.barricaded_doors:
            self.barricaded_doors.discard((r, c))
            self.grid[r, c] = DOOR
            return True
        return False

    def drop_fake_food(self, r: int, c: int) -> bool:
        if self.grid[r, c] == EMPTY:
            self.grid[r, c] = FAKE_FOOD
            self.fake_food.append((r, c))
            return True
        return False

    def drop_scent(self, r: int, c: int, strength: float = 1.0,
                   ttl: int = 15) -> None:
        self.scent_map[r, c] = strength
        self.scent_ttl[r, c] = ttl

    def consume_food(self, r: int, c: int) -> Tuple[bool, bool]:
        """Returns (consumed, was_fake)."""
        if self.grid[r, c] == FOOD:
            self.grid[r, c] = EMPTY
            return True, False
        if self.grid[r, c] == FAKE_FOOD:
            self.grid[r, c] = EMPTY
            if (r, c) in self.fake_food:
                self.fake_food.remove((r, c))
            return True, True
        return False, False

    def push_heavy_obj(self, obj_r: int, obj_c: int,
                       dr: int, dc: int) -> bool:
        """
        Move a heavy object one tile in direction (dr,dc).
        Caller is responsible for verifying 2-agent adjacency.
        """
        new_r, new_c = obj_r + dr, obj_c + dc
        if not (0 <= new_r < self.height and 0 <= new_c < self.width):
            return False
        if self.grid[new_r, new_c] not in WALKABLE:
            return False
        self.grid[obj_r, obj_c] = EMPTY
        self.grid[new_r, new_c] = HEAVY_OBJ
        idx = self.heavy_positions.index((obj_r, obj_c))
        self.heavy_positions[idx] = (new_r, new_c)
        return True

    # ── Time step ────────────────────────────────────────────────────────────

    def step_scent(self) -> None:
        """Decay scent every world step."""
        mask = self.scent_ttl > 0
        self.scent_ttl[mask]  -= 1
        self.scent_map[mask]  *= 0.85
        # Zero out fully decayed
        self.scent_map[self.scent_ttl == 0] = 0.0

    # ── Observation helpers ───────────────────────────────────────────────────

    def get_fov(self, row: int, col: int,
                radius: int = 3) -> np.ndarray:
        """
        Return a (2*radius+1, 2*radius+1) grid of tiles centred on agent.
        Tiles in dark rooms are masked to -1 (unknown).
        Tiles outside map bounds are masked to WALL.
        """
        size  = 2 * radius + 1
        fov   = np.full((size, size), WALL, dtype=np.int32)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = row + dr, col + dc
                gi, gj = dr + radius, dc + radius
                if 0 <= r < self.height and 0 <= c < self.width:
                    if not self.is_lit(r, c):
                        fov[gi, gj] = -1   # dark tile
                    else:
                        fov[gi, gj] = self.grid[r, c]
        return fov

    def get_room_light_status(self) -> List[bool]:
        return [r.light_on for r in self.rooms]

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self.grid = self._base_grid.copy()
        self.barricaded_doors.clear()
        self.scent_map[:] = 0
        self.scent_ttl[:] = 0
        self.fake_food.clear()
        for room in self.rooms:
            room.light_on = True

    def __repr__(self) -> str:
        symbols = {EMPTY:'.', WALL:'#', DOOR:'+', HIDE_SPOT:'H',
                   LIGHT_SW:'L', FOOD:'f', FAKE_FOOD:'F',
                   BARRICADE:'B', HEAVY_OBJ:'O', SCENT:'~'}
        rows = []
        for r in range(self.height):
            row = ''
            for c in range(self.width):
                row += symbols.get(self.grid[r,c], '?')
            rows.append(row)
        return '\n'.join(rows)