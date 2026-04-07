from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, state_shape: tuple[int, int, int], device: torch.device) -> None:
        self.capacity = int(capacity)
        self.device = device

        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

        self.pos = 0
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.pos] = (state * 255.0).astype(np.uint8)
        self.next_states[self.pos] = (next_state * 255.0).astype(np.uint8)
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = float(reward)
        self.dones[self.pos] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)

        states = torch.from_numpy(self.states[idx]).to(self.device, dtype=torch.float32) / 255.0
        next_states = torch.from_numpy(self.next_states[idx]).to(self.device, dtype=torch.float32) / 255.0
        actions = torch.from_numpy(self.actions[idx]).to(self.device, dtype=torch.long)
        rewards = torch.from_numpy(self.rewards[idx]).to(self.device, dtype=torch.float32)
        dones = torch.from_numpy(self.dones[idx]).to(self.device, dtype=torch.float32)

        # HWC -> CHW
        states = states.permute(0, 3, 1, 2).contiguous()
        next_states = next_states.permute(0, 3, 1, 2).contiguous()

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return self.size
