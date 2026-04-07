from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from sim2.entities import AgentEntity


@dataclass
class SimState:
    seed: int
    step: int
    max_steps: int
    prep_steps: int
    width: int
    height: int
    grid: np.ndarray
    agents: List[AgentEntity]

    def alive_hiders(self) -> List[AgentEntity]:
        return [a for a in self.agents if a.team.value == 0 and a.alive]

    def alive_seekers(self) -> List[AgentEntity]:
        return [a for a in self.agents if a.team.value == 1 and a.alive]

    def trajectory_digest(self) -> Dict[str, object]:
        return {
            "step": int(self.step),
            "grid_sum": int(self.grid.sum()),
            "agents": tuple((a.id, a.team.value, a.row, a.col, a.alive, a.direction) for a in self.agents),
        }
