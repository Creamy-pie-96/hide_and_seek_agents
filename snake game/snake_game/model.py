from __future__ import annotations

import torch
from torch import nn


class SnakeDQN(nn.Module):
    def __init__(self, input_channels: int = 3, grid_size: int = 20, n_actions: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        flattened = 64 * grid_size * grid_size
        self.head = nn.Sequential(
            nn.Linear(flattened, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [B,C,H,W] or [B,H,W,C], got {tuple(x.shape)}")

        # If HWC format is passed, convert to CHW.
        if x.shape[-1] == 3 and x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)

        x = x.float()
        return self.head(self.features(x))
