from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple


class Team(IntEnum):
    HIDER = 0
    SEEKER = 1


class Action(IntEnum):
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class Tile(IntEnum):
    EMPTY = 0
    WALL = 1
    DOOR = 2
    HIDE_SPOT = 3
    SWITCH = 4
    FOOD = 5
    BOX = 6


MOVE_DELTAS = {
    Action.STAY: (0, 0),
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}


@dataclass
class AgentEntity:
    id: int
    team: Team
    row: int
    col: int
    direction: Tuple[int, int] = (0, 1)
    alive: bool = True
    stunned_for: int = 0

    def pos(self) -> Tuple[int, int]:
        return (self.row, self.col)
