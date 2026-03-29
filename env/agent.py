"""
agent.py — Agent state, actions, field-of-view and team mechanics.

Two teams:
  HIDER  — tries to survive until time runs out.
  SEEKER — tries to tag all hiders before time runs out.

Actions (discrete, 11 total):
  0  STAY
  1  UP
  2  DOWN
  3  LEFT
  4  RIGHT
  5  TOGGLE_LIGHT     — flip light switch if standing on LIGHT_SW tile
  6  BARRICADE_DOOR   — block adjacent DOOR tile
  7  DROP_FAKE_FOOD   — place a fake food on current tile
  8  DROP_SCENT       — leave a scent trail on current tile
  9  TEAM_SIGNAL_A    — coordinate signal A (used for team-only mechanics)
  10 TEAM_SIGNAL_B    — coordinate signal B

Team-only mechanics (checked in environment step):
  • Heavy barricade push: 2+ hiders adjacent to HEAVY_OBJ, both signal A
  • Coordinated sweep:    2+ seekers enter same room from different doors
  • Full blackout:        all 3 hiders toggle lights simultaneously
"""

from __future__ import annotations
import numpy as np
from enum import IntEnum
from typing import Tuple, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from env.world import World

# ── Enums ────────────────────────────────────────────────────────────────────

class Team(IntEnum):
    HIDER  = 0
    SEEKER = 1

class Action(IntEnum):
    STAY          = 0
    UP            = 1
    DOWN          = 2
    LEFT          = 3
    RIGHT         = 4
    TOGGLE_LIGHT  = 5
    BARRICADE     = 6
    DROP_FAKE_FOOD= 7
    DROP_SCENT    = 8
    SIGNAL_A      = 9
    SIGNAL_B      = 10

N_ACTIONS = len(Action)

MOVE_DELTAS = {
    Action.UP:    (-1,  0),
    Action.DOWN:  ( 1,  0),
    Action.LEFT:  ( 0, -1),
    Action.RIGHT: ( 0,  1),
    Action.STAY:  ( 0,  0),
}

# ── FOV & observation constants ──────────────────────────────────────────────
FOV_RADIUS   = 4      # tiles in each direction
FOV_SIZE     = 2 * FOV_RADIUS + 1   # 9×9 window
N_TILE_TYPES = 11     # tile int values 0-10 (including -1 mapped to 10 for dark)

# Observation layout (flat):
#   [fov_grid (FOV_SIZE²),  own_state (6),  teammate_states (2 × 5),  team_signals (2)]
#   = 81 + 6 + 10 + 2 = 99 floats
OBS_FOV       = FOV_SIZE * FOV_SIZE   # 81
OBS_SELF      = 6    # row, col, dir, alive, food_count, cooldown
OBS_TEAMMATE  = 5    # row, col, alive, food_count, last_signal
OBS_SIGNAL    = 2    # last signals from self
OBS_DIM = OBS_FOV + OBS_SELF + 2 * OBS_TEAMMATE + OBS_SIGNAL  # 99


class Agent:
    """
    Single agent in the hide-and-seek environment.

    Parameters
    ----------
    agent_id : unique int (0-5, where 0-2 are hiders, 3-5 seekers)
    team     : Team.HIDER or Team.SEEKER
    """

    def __init__(self, agent_id: int, team: Team):
        self.agent_id  = agent_id
        self.team      = team

        # Position & movement
        self.row: int  = 0
        self.col: int  = 0
        self.last_dir: Tuple[int,int] = (0, 1)   # facing right initially

        # Vitals
        self.alive: bool = True
        self.food_count: int = 0
        self.caught: bool    = False    # hider got tagged by seeker

        # Cooldowns (steps before an action can fire again)
        self.barricade_cooldown: int  = 0
        self.fake_food_cooldown: int  = 0
        self.scent_cooldown:     int  = 0
        self.light_cooldown:     int  = 0

        # Team coordination
        self.last_signal: int = 0       # 0=none, 1=signal_A, 2=signal_B
        self.signal_A_this_step = False
        self.signal_B_this_step = False

        # Penalty tracking (for reward shaping)
        self.steps_alive: int = 0
        self.rooms_visited: set = set()

        # Stun (seekers can stun hiders briefly via coordinated sweep)
        self.stunned_for: int = 0

    # ── Spawn ────────────────────────────────────────────────────────────────

    def spawn(self, row: int, col: int) -> None:
        self.row, self.col = row, col
        self.alive         = True
        self.caught        = False
        self.food_count    = 0
        self.steps_alive   = 0
        self.rooms_visited = set()
        self.last_signal   = 0
        self.stunned_for   = 0
        self._reset_cooldowns()

    def _reset_cooldowns(self) -> None:
        self.barricade_cooldown  = 0
        self.fake_food_cooldown  = 0
        self.scent_cooldown      = 0
        self.light_cooldown      = 0

    # ── Core step ────────────────────────────────────────────────────────────

    def step(self, action: int, world: 'World') -> dict:
        """
        Apply an action. Returns an info dict describing what happened.
        Called by the environment AFTER action validity checks.
        """
        if not self.alive or self.stunned_for > 0:
            if self.stunned_for > 0:
                self.stunned_for -= 1
            return {'moved': False, 'action': action}

        act = Action(action)
        info = {'moved': False, 'ate_food': False, 'ate_fake': False,
                'toggled_light': False, 'barricaded': False,
                'dropped_fake': False, 'dropped_scent': False,
                'signal_a': False, 'signal_b': False, 'action': action}

        # Decay cooldowns
        self.barricade_cooldown  = max(0, self.barricade_cooldown  - 1)
        self.fake_food_cooldown  = max(0, self.fake_food_cooldown  - 1)
        self.scent_cooldown      = max(0, self.scent_cooldown      - 1)
        self.light_cooldown      = max(0, self.light_cooldown      - 1)

        # Clear per-step signals
        self.signal_A_this_step = False
        self.signal_B_this_step = False

        if act in MOVE_DELTAS:
            dr, dc = MOVE_DELTAS[act]
            new_r, new_c = self.row + dr, self.col + dc
            if act != Action.STAY and world.is_walkable(new_r, new_c):
                self.row, self.col = new_r, new_c
                if dr != 0 or dc != 0:
                    self.last_dir = (dr, dc)
                info['moved'] = True

                # Auto-consume food on new tile
                consumed, was_fake = world.consume_food(self.row, self.col)
                if consumed:
                    if not was_fake:
                        self.food_count += 1
                        info['ate_food'] = True
                    else:
                        info['ate_fake'] = True  # penalty for seeker

        elif act == Action.TOGGLE_LIGHT:
            if self.light_cooldown == 0:
                toggled = world.toggle_light(self.row, self.col)
                if toggled:
                    self.light_cooldown = 5
                    info['toggled_light'] = True

        elif act == Action.BARRICADE:
            if self.barricade_cooldown == 0:
                # Try adjacent door tiles
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    r, c = self.row + dr, self.col + dc
                    if world.place_barricade(r, c):
                        self.barricade_cooldown = 10
                        info['barricaded'] = True
                        break

        elif act == Action.DROP_FAKE_FOOD:
            if self.fake_food_cooldown == 0 and self.team == Team.HIDER:
                if world.drop_fake_food(self.row, self.col):
                    self.fake_food_cooldown = 8
                    info['dropped_fake'] = True

        elif act == Action.DROP_SCENT:
            if self.scent_cooldown == 0:
                world.drop_scent(self.row, self.col)
                self.scent_cooldown = 3
                info['dropped_scent'] = True

        elif act == Action.SIGNAL_A:
            self.signal_A_this_step = True
            self.last_signal = 1
            info['signal_a'] = True

        elif act == Action.SIGNAL_B:
            self.signal_B_this_step = True
            self.last_signal = 2
            info['signal_b'] = True

        self.steps_alive += 1

        # Track room visits
        room = world.get_room(self.row, self.col)
        if room:
            self.rooms_visited.add(room.room_id)

        return info

    # ── Observation ──────────────────────────────────────────────────────────

    def get_observation(self, world: 'World',
                        teammates: List['Agent']) -> np.ndarray:
        """
        Build the flat observation vector for this agent.
        Shape: (OBS_DIM,) = (99,) float32
        """
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        ptr = 0

        # -- FOV grid (flattened, normalised) --------------------------------
        fov = world.get_fov(self.row, self.col, radius=FOV_RADIUS)
        # Map -1 (dark) to N_TILE_TYPES-1 so range is [0, N_TILE_TYPES-1]
        fov_norm = np.where(fov == -1, N_TILE_TYPES - 1, fov).astype(np.float32)
        fov_norm /= (N_TILE_TYPES - 1)   # [0, 1]
        obs[ptr:ptr + OBS_FOV] = fov_norm.flatten()
        ptr += OBS_FOV

        # -- Own state --------------------------------------------------------
        obs[ptr]   = self.row  / world.height
        obs[ptr+1] = self.col  / world.width
        obs[ptr+2] = (self.last_dir[0] + 1) / 2.0   # map (-1,0,1) to (0,0.5,1)
        obs[ptr+3] = (self.last_dir[1] + 1) / 2.0
        obs[ptr+4] = float(self.alive)
        obs[ptr+5] = min(self.food_count / 10.0, 1.0)
        ptr += OBS_SELF

        # -- Teammate states --------------------------------------------------
        for tm in teammates[:2]:   # always exactly 2 teammates for 3v3
            obs[ptr]   = tm.row  / world.height   if tm.alive else -1.0
            obs[ptr+1] = tm.col  / world.width    if tm.alive else -1.0
            obs[ptr+2] = float(tm.alive)
            obs[ptr+3] = min(tm.food_count / 10.0, 1.0)
            obs[ptr+4] = tm.last_signal / 2.0     # 0, 0.5, or 1.0
            ptr += OBS_TEAMMATE

        # -- Team signals -----------------------------------------------------
        obs[ptr]   = float(self.signal_A_this_step)
        obs[ptr+1] = float(self.signal_B_this_step)

        return obs

    # ── Utility ──────────────────────────────────────────────────────────────

    def distance_to(self, other: 'Agent') -> float:
        return abs(self.row - other.row) + abs(self.col - other.col)

    def is_adjacent_to(self, other: 'Agent') -> bool:
        return self.distance_to(other) <= 1

    def can_see(self, other: 'Agent', world: 'World') -> bool:
        """
        Agent can see another if:
        - Other is within FOV radius
        - The tile is lit
        - Other is not on a HIDE_SPOT (unless we are adjacent)
        """
        from env.world import HIDE_SPOT
        if not other.alive:
            return False
        dist = self.distance_to(other)
        if dist > FOV_RADIUS:
            return False
        if not world.is_lit(other.row, other.col):
            return False
        # Hiders in hide spots are invisible unless adjacent
        if world.grid[other.row, other.col] == HIDE_SPOT and dist > 1:
            return False
        return True

    @property
    def pos(self) -> Tuple[int, int]:
        return (self.row, self.col)

    def __repr__(self) -> str:
        team_str = "H" if self.team == Team.HIDER else "S"
        status   = "alive" if self.alive else "dead"
        return f"Agent({team_str}{self.agent_id}, ({self.row},{self.col}), {status}, food={self.food_count})"