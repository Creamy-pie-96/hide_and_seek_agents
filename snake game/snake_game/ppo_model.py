from __future__ import annotations

import torch
from torch import nn


class SnakeActorCritic(nn.Module):
    def __init__(self, input_channels: int = 3, grid_size: int = 20, n_actions: int = 3, aux_dim: int = 0) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.aux_dim = int(aux_dim)
        self.input_channels = int(input_channels)
        self._scent_start = 7
        self._scent_count = 3

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # Grid-size agnostic trunk so the same policy can operate on 10x10 and 20x20.
        self.pool = nn.AdaptiveAvgPool2d((5, 5))
        self.flatten = nn.Flatten()

        flattened = 64 * 5 * 5
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
        self.scent_policy_head = nn.Linear(self._scent_count, n_actions)

    def forward(self, x: torch.Tensor, aux: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W] or [B,H,W,C], got {tuple(x.shape)}")

        # Accept NHWC or NCHW depending on caller.
        if x.shape[-1] == self.input_channels and x.shape[1] != self.input_channels:
            x = x.permute(0, 3, 1, 2)

        x = x.float()
        feat = self.encoder(x)
        z = self.shared(self.flatten(self.pool(feat)))

        scent_logits = None
        if self.aux_dim > 0:
            if aux is None:
                aux = torch.zeros((z.shape[0], self.aux_dim), device=z.device, dtype=z.dtype)
            else:
                aux = aux.float()
            if self.aux_dim >= (self._scent_start + self._scent_count):
                scent = aux[:, self._scent_start : self._scent_start + self._scent_count]
                scent_logits = self.scent_policy_head(scent)
            z = torch.cat([z, aux], dim=-1)
        z = self.fusion(z)
        logits = self.policy_head(z)
        if scent_logits is not None:
            # Direct policy pathway from scent features to action logits.
            logits = logits + scent_logits
        value = self.value_head(z).squeeze(-1)
        return logits, value
