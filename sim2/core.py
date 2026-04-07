from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional, Tuple
import math

import numpy as np

from sim2.entities import Action, AgentEntity, MOVE_DELTAS, Team, Tile
from sim2.state import SimState
from sim2.worldgen import default_spawns, generate_layout


_WALKABLE = {
    int(Tile.EMPTY),
    int(Tile.DOOR),
    int(Tile.HIDE_SPOT),
    int(Tile.SWITCH),
    int(Tile.FOOD),
    int(Tile.BOX),
}


class PrimitiveHideSeekSim:
    """Model-free, deterministic hide-and-seek simulator core (M1)."""

    def __init__(self, width: int = 24, height: int = 24, max_steps: int = 300, prep_steps: int = 20):
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.prep_steps = prep_steps
        self.state: Optional[SimState] = None

    def reset(self, seed: int = 0) -> Dict:
        grid = generate_layout(self.width, self.height, seed)
        hider_spawns, seeker_spawns = default_spawns(grid)

        agents: List[AgentEntity] = []
        for i, (r, c) in enumerate(hider_spawns):
            agents.append(AgentEntity(id=i, team=Team.HIDER, row=r, col=c))
        for i, (r, c) in enumerate(seeker_spawns, start=3):
            agents.append(AgentEntity(id=i, team=Team.SEEKER, row=r, col=c, direction=(0, -1)))

        self.state = SimState(
            seed=int(seed),
            step=0,
            max_steps=self.max_steps,
            prep_steps=self.prep_steps,
            width=self.width,
            height=self.height,
            grid=grid,
            agents=agents,
        )
        return self.get_state()

    def _can_move(self, row: int, col: int) -> bool:
        assert self.state is not None
        if row < 0 or row >= self.state.height or col < 0 or col >= self.state.width:
            return False
        return int(self.state.grid[row, col]) in _WALKABLE

    def _line_clear(self, src: Tuple[int, int], dst: Tuple[int, int]) -> bool:
        """Bresenham-like wall check for LOS."""
        assert self.state is not None
        r0, c0 = src
        r1, c1 = dst

        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dr - dc

        r, c = r0, c0
        while (r, c) != (r1, c1):
            if (r, c) != (r0, c0) and int(self.state.grid[r, c]) == int(Tile.WALL):
                return False
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc
        return int(self.state.grid[r1, c1]) != int(Tile.WALL)

    @staticmethod
    def _within_cone(src: AgentEntity, dst: AgentEntity, fov_deg: float = 90.0, max_dist: float = 7.0) -> bool:
        dr = dst.row - src.row
        dc = dst.col - src.col
        dist = math.sqrt(dr * dr + dc * dc)
        if dist > max_dist:
            return False
        if dist < 1e-6:
            return True

        dir_r, dir_c = src.direction
        dir_norm = math.sqrt(dir_r * dir_r + dir_c * dir_c)
        if dir_norm < 1e-6:
            return True

        v1r, v1c = dir_r / dir_norm, dir_c / dir_norm
        v2r, v2c = dr / dist, dc / dist
        dot = max(-1.0, min(1.0, v1r * v2r + v1c * v2c))
        angle = math.degrees(math.acos(dot))
        return angle <= (fov_deg * 0.5)

    def _apply_visibility_and_catches(self) -> None:
        assert self.state is not None
        for seeker in self.state.alive_seekers():
            for hider in self.state.alive_hiders():
                if not self._within_cone(seeker, hider):
                    continue
                if not self._line_clear(seeker.pos(), hider.pos()):
                    continue
                hider.alive = False

    def step(self, actions: Dict[int, int]) -> Tuple[Dict, Dict[int, float], Dict[str, bool], Dict]:
        assert self.state is not None, "Call reset() before step()"

        self.state.step += 1
        is_prep = self.state.step <= self.state.prep_steps

        rewards = {a.id: 0.0 for a in self.state.agents}
        info: Dict[str, object] = {"prep": is_prep}

        # Movement and collision.
        for agent in self.state.agents:
            if not agent.alive:
                continue
            if is_prep and agent.team == Team.SEEKER:
                continue

            act = Action(int(actions.get(agent.id, int(Action.STAY))))
            dr, dc = MOVE_DELTAS[act]
            if (dr, dc) != (0, 0):
                agent.direction = (dr, dc)
            nr, nc = agent.row + dr, agent.col + dc
            if self._can_move(nr, nc):
                agent.row, agent.col = nr, nc

        # Visibility and catches only after prep.
        if not is_prep:
            before_alive = len(self.state.alive_hiders())
            self._apply_visibility_and_catches()
            after_alive = len(self.state.alive_hiders())
            caught = before_alive - after_alive
            if caught > 0:
                for a in self.state.agents:
                    if a.team == Team.SEEKER and a.alive:
                        rewards[a.id] += float(caught)
                    if a.team == Team.HIDER and not a.alive:
                        rewards[a.id] -= 1.0

        # Baseline team rewards.
        for a in self.state.agents:
            if not a.alive:
                continue
            rewards[a.id] += 0.1 if a.team == Team.HIDER else -0.05

        all_hiders_caught = len(self.state.alive_hiders()) == 0
        time_up = self.state.step >= self.state.max_steps
        done = all_hiders_caught or time_up

        if done:
            if all_hiders_caught:
                for a in self.state.agents:
                    if a.team == Team.SEEKER and a.alive:
                        rewards[a.id] += 2.0
            else:
                for a in self.state.agents:
                    if a.team == Team.HIDER and a.alive:
                        rewards[a.id] += 2.0

        done_dict = {"__all__": done}
        info["alive_hiders"] = len(self.state.alive_hiders())
        info["alive_seekers"] = len(self.state.alive_seekers())
        return self.get_state(), rewards, done_dict, info

    def get_state(self) -> Dict:
        assert self.state is not None
        return {
            "seed": self.state.seed,
            "step": self.state.step,
            "max_steps": self.state.max_steps,
            "prep": self.state.step <= self.state.prep_steps,
            "width": self.state.width,
            "height": self.state.height,
            "grid": self.state.grid.copy(),
            "agents": [
                {
                    "id": a.id,
                    "team": int(a.team),
                    "row": a.row,
                    "col": a.col,
                    "alive": a.alive,
                    "stunned": bool(a.stunned_for > 0),
                    "dir": [a.direction[0], a.direction[1]],
                }
                for a in self.state.agents
            ],
        }

    def get_render_state(self) -> Dict:
        """Compatibility state for current frame and video pipeline."""
        s = self.get_state()
        grid = s["grid"]
        return {
            "grid": grid,
            "scent_map": np.zeros_like(grid, dtype=np.float32),
            "rooms": [],
            "tile_to_room": {},
            "light_ping": None,
            "agents": [
                (a["id"], a["row"], a["col"], a["team"], a["alive"], 0, a["stunned"]) for a in s["agents"]
            ],
            "step": s["step"],
            "max_steps": s["max_steps"],
            "prep": s["prep"],
            "hiders_caught": sum(1 for a in s["agents"] if a["team"] == int(Team.HIDER) and not a["alive"]),
        }

    def get_serializable_render_state(self) -> Dict:
        rs = self.get_render_state()
        return {
            "grid": rs["grid"].astype(int).tolist(),
            "scent_map": rs["scent_map"].astype(float).tolist(),
            "room_lights": {},
            "tile_to_room": [],
            "light_ping": None,
            "agents": [
                {
                    "id": int(aid),
                    "row": int(row),
                    "col": int(col),
                    "team": int(team),
                    "alive": bool(alive),
                    "food": int(food),
                    "stunned": bool(stunned),
                }
                for (aid, row, col, team, alive, food, stunned) in rs["agents"]
            ],
            "step": int(rs["step"]),
            "max_steps": int(rs["max_steps"]),
            "prep": bool(rs["prep"]),
            "hiders_caught": int(rs["hiders_caught"]),
            "width": int(self.width),
            "height": int(self.height),
        }

    def trajectory_digest(self) -> Dict[str, object]:
        assert self.state is not None
        return self.state.trajectory_digest()
