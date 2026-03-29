"""
network.py — Shared-weight Actor-Critic for MAPPO.

Architecture per agent:
  1. CNN  — encodes the 9×9 FOV grid into spatial features
  2. MLP  — encodes scalar observations (own state + teammate states + signals)
  3. Cat  — concatenate CNN output + MLP output
  4. LSTM — adds temporal memory (agents remember where they've searched)
  5. Heads:
       • Policy head → logits over N_ACTIONS (softmax → sample action)
       • Value  head → scalar state-value estimate

Weights are SHARED across all agents on the same team (parameter-sharing
MAPPO). In 3v3 we have:
    hider_net  : one ActorCritic used by all 3 hiders
    seeker_net : one ActorCritic used by all 3 seekers
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

from env.agent import (OBS_DIM, OBS_FOV, OBS_SELF, OBS_TEAMMATE,
                       N_ACTIONS, FOV_SIZE, N_TILE_TYPES)

# ── Sizes ────────────────────────────────────────────────────────────────────
CNN_CHANNELS  = [1, 16, 32]      # input channels, hidden, output channels
CNN_OUT_DIM   = 32 * 3 * 3      # after 2×MaxPool on 9×9 → 2×2... actually 3×3

SCALAR_DIM    = OBS_DIM - OBS_FOV          # 99 - 81 = 18
SCALAR_HIDDEN = 64

LSTM_HIDDEN   = 128
LSTM_LAYERS   = 1

POLICY_HIDDEN = 64
VALUE_HIDDEN  = 64


class FOVEncoder(nn.Module):
    """
    Small CNN that encodes the 9×9 FOV grid.
    Input : (batch, 1, 9, 9)  — single channel, tile values normalised [0,1]
    Output: (batch, CNN_OUT_DIM)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # → (16, 9, 9)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # → (32, 9, 9)
            nn.ReLU(),
            nn.MaxPool2d(3),                               # → (32, 3, 3)
            nn.Flatten(),                                  # → 288
        )
        self.out_dim = 32 * 3 * 3   # 288

    def forward(self, fov: torch.Tensor) -> torch.Tensor:
        # fov: (B, FOV_SIZE²) → (B, 1, FOV_SIZE, FOV_SIZE)
        B = fov.shape[0]
        x = fov.view(B, 1, FOV_SIZE, FOV_SIZE)
        return self.net(x)


class ScalarEncoder(nn.Module):
    """MLP for non-spatial scalars (own state, teammates, signals)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(SCALAR_DIM, SCALAR_HIDDEN),
            nn.LayerNorm(SCALAR_HIDDEN),
            nn.ReLU(),
            nn.Linear(SCALAR_HIDDEN, SCALAR_HIDDEN),
            nn.ReLU(),
        )
        self.out_dim = SCALAR_HIDDEN

    def forward(self, scalars: torch.Tensor) -> torch.Tensor:
        return self.net(scalars)


class ActorCritic(nn.Module):
    """
    Full actor-critic module with LSTM core.

    Forward call takes (obs, lstm_state) and returns:
        (action_logits, value, new_lstm_state)

    LSTM state shape: (num_layers, batch, hidden) for both h and c.
    Pass hidden=None on episode start to auto-initialise to zeros.
    """

    def __init__(self, n_actions: int = N_ACTIONS):
        super().__init__()
        self.fov_enc    = FOVEncoder()
        self.scalar_enc = ScalarEncoder()

        combined_dim    = self.fov_enc.out_dim + self.scalar_enc.out_dim  # 288+64=352

        self.lstm = nn.LSTM(
            input_size   = combined_dim,
            hidden_size  = LSTM_HIDDEN,
            num_layers   = LSTM_LAYERS,
            batch_first  = True,
        )

        # Policy head
        self.policy = nn.Sequential(
            nn.Linear(LSTM_HIDDEN, POLICY_HIDDEN),
            nn.ReLU(),
            nn.Linear(POLICY_HIDDEN, n_actions),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(LSTM_HIDDEN, VALUE_HIDDEN),
            nn.ReLU(),
            nn.Linear(VALUE_HIDDEN, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Policy head: smaller init for more uniform initial policy
        nn.init.orthogonal_(self.policy[-1].weight, gain=0.01)

    def forward(
        self,
        obs: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor,
               Tuple[torch.Tensor, torch.Tensor]]:
        """
        obs        : (B, OBS_DIM) float32
        lstm_state : ((LAYERS, B, HIDDEN), (LAYERS, B, HIDDEN)) or None

        Returns:
            logits     : (B, N_ACTIONS)
            value      : (B,)
            lstm_state : updated hidden state
        """
        B = obs.shape[0]

        fov_part    = obs[:, :OBS_FOV]                    # (B, 81)
        scalar_part = obs[:, OBS_FOV:]                    # (B, 18)

        fov_feat    = self.fov_enc(fov_part)              # (B, 288)
        scalar_feat = self.scalar_enc(scalar_part)        # (B, 64)

        combined = torch.cat([fov_feat, scalar_feat], dim=-1)  # (B, 352)

        # LSTM expects (B, T, input_dim); we run one step at a time
        lstm_in = combined.unsqueeze(1)                   # (B, 1, 352)

        if lstm_state is None:
            lstm_state = self._init_hidden(B, obs.device)

        lstm_out, new_state = self.lstm(lstm_in, lstm_state)  # (B,1,128)
        h = lstm_out.squeeze(1)                                # (B, 128)

        logits = self.policy(h)                           # (B, N_ACTIONS)
        value  = self.value_head(h).squeeze(-1)           # (B,)

        return logits, value, new_state

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        lstm_state: Optional[Tuple] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, Tuple]:
        """
        Used during rollout collection (action=None → sample)
        and training (action provided → evaluate log_prob + entropy).

        Returns: action, log_prob, entropy, value, new_lstm_state
        """
        logits, value, new_state = self.forward(obs, lstm_state)
        dist     = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()

        return action, log_prob, entropy, value, new_state

    def _init_hidden(
        self, batch: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(LSTM_LAYERS, batch, LSTM_HIDDEN, device=device)
        c = torch.zeros(LSTM_LAYERS, batch, LSTM_HIDDEN, device=device)
        return (h, c)

    def init_hidden(self, batch: int = 1,
                    device: Optional[torch.device] = None) -> Tuple:
        if device is None:
            device = next(self.parameters()).device
        return self._init_hidden(batch, device)


# ── Factory ───────────────────────────────────────────────────────────────────

def make_networks(device: str = "cpu") -> Tuple[ActorCritic, ActorCritic]:
    """Return (hider_net, seeker_net) ready to train."""
    dev = torch.device(device)
    hider_net  = ActorCritic(N_ACTIONS).to(dev)
    seeker_net = ActorCritic(N_ACTIONS).to(dev)
    return hider_net, seeker_net


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)