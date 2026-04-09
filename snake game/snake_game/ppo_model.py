from __future__ import annotations

import torch
from torch import nn


class SnakeActorCritic(nn.Module):
    def __init__(self, input_channels: int = 3, grid_size: int = 20, n_actions: int = 3, aux_dim: int = 0) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.aux_dim = int(aux_dim)
        self.input_channels = int(input_channels)

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        flattened = 64 * grid_size * grid_size
        self.shared = nn.Sequential(
            nn.Linear(flattened, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

        fusion_in = 128 + self.aux_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 128),
            nn.ReLU(inplace=True),
        )

        self.policy_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor, aux: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W] or [B,H,W,C], got {tuple(x.shape)}")

        # Accept NHWC or NCHW depending on caller.
        if x.shape[-1] == self.input_channels and x.shape[1] != self.input_channels:
            x = x.permute(0, 3, 1, 2)

        x = x.float()
        z = self.shared(self.encoder(x))
        if self.aux_dim > 0:
            if aux is None:
                aux = torch.zeros((z.shape[0], self.aux_dim), device=z.device, dtype=z.dtype)
            else:
                aux = aux.float()
            z = torch.cat([z, aux], dim=-1)
        z = self.fusion(z)
        logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)
        return logits, value
