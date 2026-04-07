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
from typing import Tuple, Optional, cast

from env.agent import (OBS_DIM, OBS_FOV, OBS_SELF, OBS_TEAMMATE,
                       N_ACTIONS, FOV_SIZE, N_TILE_TYPES, FOV_CHANNELS,
                       GLOBAL_STATE_DIM)

# ── Sizes ────────────────────────────────────────────────────────────────────
CNN_CHANNELS  = [FOV_CHANNELS, 16, 32]
CNN_OUT_DIM   = 32 * 3 * 3      # after 2×MaxPool on 9×9 → 2×2... actually 3×3

SCALAR_DIM    = OBS_DIM - OBS_FOV
SCALAR_HIDDEN = 128
TEAM_SIZE     = 3
VALUE_OBS_DIM = GLOBAL_STATE_DIM

LSTM_HIDDEN   = 256
LSTM_LAYERS   = 1

POLICY_HIDDEN = 128
VALUE_HIDDEN  = 128


class FOVEncoder(nn.Module):
    """
    Small CNN that encodes the 9×9 FOV grid.
    Input : (batch, 1, 9, 9)  — single channel, tile values normalised [0,1]
    Output: (batch, CNN_OUT_DIM)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(FOV_CHANNELS, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Flatten(),
        )
        self.out_dim = 32 * 3 * 3

    def forward(self, fov: torch.Tensor) -> torch.Tensor:
        # fov: flattened interleaved [tile, scent] channels.
        # reshape: (B, FOV_SIZE, FOV_SIZE, C) -> (B, C, FOV_SIZE, FOV_SIZE)
        B = fov.shape[0]
        x = fov.view(B, FOV_SIZE, FOV_SIZE, FOV_CHANNELS).permute(0, 3, 1, 2)
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

        # Centralized critic path (team context)
        self.value_encoder = nn.Sequential(
            nn.Linear(VALUE_OBS_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, VALUE_HIDDEN),
            nn.ReLU(),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(VALUE_HIDDEN, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                w = getattr(m, "weight", None)
                b = getattr(m, "bias", None)
                if isinstance(w, torch.Tensor):
                    nn.init.orthogonal_(w, gain=np.sqrt(2))
                if isinstance(b, torch.Tensor):
                    nn.init.zeros_(b)
        # Policy head: smaller init for more uniform initial policy
        policy_last = cast(nn.Linear, self.policy[-1])
        nn.init.orthogonal_(policy_last.weight, gain=0.01)

    def forward(
        self,
        obs: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        value_obs: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor,
               Tuple[torch.Tensor, torch.Tensor]]:
        """
        obs        : (B, OBS_DIM) float32
        lstm_state : ((LAYERS, B, HIDDEN), (LAYERS, B, HIDDEN)) or None

        Returns:
            logits     : (B, N_ACTIONS)
            value      : (B,) computed from centralized value input
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
        logits = self._apply_action_mask(logits, action_mask)
        if value_obs is None:
            value_obs = torch.zeros((B, VALUE_OBS_DIM), dtype=obs.dtype, device=obs.device)
        vfeat = self.value_encoder(value_obs)
        value  = self.value_head(vfeat).squeeze(-1)

        return logits, value, new_state

    def _apply_action_mask(self, logits: torch.Tensor, action_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if action_mask is None:
            return logits
        # action_mask: 1.0 for valid, 0.0 for invalid
        return logits.masked_fill(action_mask <= 0.0, -1e10)

    def forward_sequence(
        self,
        obs_seq: torch.Tensor,
        reset_seq: torch.Tensor,
        value_obs_seq: Optional[torch.Tensor] = None,
        action_mask_seq: Optional[torch.Tensor] = None,
        init_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Recurrent unroll over sequence.

        obs_seq: (B, T, OBS_DIM)
        reset_seq: (B, T) bool, True means reset hidden before this step.
        value_obs_seq: (B, T, VALUE_OBS_DIM) optional
        action_mask_seq: (B, T, N_ACTIONS) optional
        """
        B, T, _ = obs_seq.shape
        if init_state is None:
            state = self._init_hidden(B, obs_seq.device)
        else:
            state = init_state

        logits_steps = []
        value_steps = []
        for t in range(T):
            rst = reset_seq[:, t].view(1, B, 1).to(dtype=state[0].dtype)
            h, c = state
            h = h * (1.0 - rst)
            c = c * (1.0 - rst)
            state = (h, c)

            obs_t = obs_seq[:, t, :]
            vobs_t = None if value_obs_seq is None else value_obs_seq[:, t, :]
            amask_t = None if action_mask_seq is None else action_mask_seq[:, t, :]
            logits_t, value_t, state = self.forward(
                obs_t,
                lstm_state=state,
                value_obs=vobs_t,
                action_mask=amask_t,
            )
            logits_steps.append(logits_t)
            value_steps.append(value_t)

        logits_seq = torch.stack(logits_steps, dim=1)
        value_seq = torch.stack(value_steps, dim=1)
        return logits_seq, value_seq, state

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        lstm_state: Optional[Tuple] = None,
        action: Optional[torch.Tensor] = None,
        value_obs: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, Tuple]:
        """
        Used during rollout collection (action=None → sample)
        and training (action provided → evaluate log_prob + entropy).

        Returns: action, log_prob, entropy, value, new_lstm_state
        """
        logits, value, new_state = self.forward(
            obs,
            lstm_state,
            value_obs=value_obs,
            action_mask=action_mask,
        )
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