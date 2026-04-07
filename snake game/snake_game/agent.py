from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .model import SnakeDQN


@dataclass
class AgentConfig:
    lr: float = 1e-4
    gamma: float = 0.99
    grad_clip: float = 10.0
    target_update_steps: int = 1000


class DQNAgent:
    def __init__(self, grid_size: int, device: torch.device, cfg: AgentConfig) -> None:
        self.device = device
        self.cfg = cfg
        self.n_actions = 3

        self.online = SnakeDQN(grid_size=grid_size, n_actions=self.n_actions).to(device)
        self.target = SnakeDQN(grid_size=grid_size, n_actions=self.n_actions).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optim = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.train_steps = 0

    @torch.no_grad()
    def act(self, state: np.ndarray, epsilon: float) -> int:
        if np.random.random() < epsilon:
            return int(np.random.randint(0, self.n_actions))

        s = torch.from_numpy(state).to(self.device, dtype=torch.float32).unsqueeze(0)
        s = s.permute(0, 3, 1, 2).contiguous()
        q = self.online(s)
        return int(torch.argmax(q, dim=1).item())

    def optimize(self, batch) -> float:
        states, actions, rewards, next_states, dones = batch

        q_values = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_online_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            next_target_q = self.target(next_states).gather(1, next_online_actions).squeeze(1)
            targets = rewards + self.cfg.gamma * (1.0 - dones) * next_target_q

        loss = F.smooth_l1_loss(q_values, targets)

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.optim.step()

        self.train_steps += 1
        if self.train_steps % self.cfg.target_update_steps == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())

    def save(self, path: str, extra: dict | None = None) -> None:
        payload = {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optim": self.optim.state_dict(),
            "train_steps": self.train_steps,
            "cfg": self.cfg.__dict__,
            "extra": extra or {},
        }
        torch.save(payload, path)

    def load(self, path: str, strict: bool = True) -> dict:
        payload = torch.load(path, map_location=self.device)
        if "online" in payload:
            self.online.load_state_dict(payload["online"], strict=strict)
            if "target" in payload:
                self.target.load_state_dict(payload["target"], strict=strict)
            else:
                self.target.load_state_dict(payload["online"], strict=strict)
            if "optim" in payload:
                self.optim.load_state_dict(payload["optim"])
            self.train_steps = int(payload.get("train_steps", 0))
            return payload.get("extra", {})

        # Fallback: plain state dict
        self.online.load_state_dict(payload, strict=strict)
        self.target.load_state_dict(payload, strict=strict)
        return {}
