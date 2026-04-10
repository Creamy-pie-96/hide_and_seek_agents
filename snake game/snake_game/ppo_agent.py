from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from .ppo_model import SnakeActorCritic


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    rollout_steps: int = 256
    n_envs: int = 8
    update_epochs: int = 4
    minibatch_size: int = 512
    aux_dim: int = 14
    obs_channels: int = 3


class PPOAgent:
    def __init__(self, grid_size: int, device: torch.device, cfg: PPOConfig) -> None:
        self.device = device
        self.cfg = cfg
        self.net = SnakeActorCritic(
            input_channels=cfg.obs_channels,
            grid_size=grid_size,
            n_actions=3,
            aux_dim=cfg.aux_dim,
        ).to(device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.update_count = 0

    @staticmethod
    def _apply_action_mask(logits: torch.Tensor, aux: torch.Tensor | None) -> torch.Tensor:
        if aux is None or aux.shape[-1] < 3:
            return logits
        invalid = aux[:, :3] > 0.5  # [B,3]: danger_front/right/left
        all_invalid = invalid.all(dim=1, keepdim=True)
        invalid = invalid & (~all_invalid)
        masked = logits.masked_fill(invalid, -1e9)

        # Food-direction action guidance (mask only near food to avoid over-constraining exploration).
        if aux.shape[-1] < 14:
            return masked

        food_up = aux[:, 3] > 0.5
        food_down = aux[:, 4] > 0.5
        food_right = aux[:, 5] > 0.5
        food_left = aux[:, 6] > 0.5
        scent_intensity = aux[:, 9]  # 1 / (manhattan + 1)
        heading = torch.argmax(aux[:, 10:14], dim=1)  # 0=up,1=right,2=down,3=left

        # Preferred absolute directions toward food.
        pref_abs = torch.zeros((aux.shape[0], 4), dtype=torch.bool, device=aux.device)
        pref_abs[:, 0] = food_up
        pref_abs[:, 1] = food_right
        pref_abs[:, 2] = food_down
        pref_abs[:, 3] = food_left

        rel_pref = torch.zeros((aux.shape[0], 3), dtype=torch.bool, device=aux.device)
        for rel_action in range(3):
            abs_dir = (heading + (1 if rel_action == 1 else (-1 if rel_action == 2 else 0))) % 4
            rel_pref[:, rel_action] = pref_abs.gather(1, abs_dir.unsqueeze(1)).squeeze(1)

        safe = ~invalid
        near_food = scent_intensity >= 0.20  # dist <= 4
        preferred_safe = rel_pref & safe
        preferred_safe_count = preferred_safe.sum(dim=1)

        # Hard-mask to the single preferred safe action in close-range strike zone.
        force_strike = near_food & (preferred_safe_count == 1)
        if torch.any(force_strike):
            strike_mask = safe.clone()
            strike_mask[force_strike] = ~preferred_safe[force_strike]
            all_invalid_strike = strike_mask.all(dim=1, keepdim=True)
            strike_mask = strike_mask & (~all_invalid_strike)
            masked = masked.masked_fill(strike_mask, -1e9)

        # Mild bias toward preferred safe actions when food is moderately near.
        mid_food = (scent_intensity >= 0.12) & (preferred_safe_count >= 1)
        if torch.any(mid_food):
            bias = torch.zeros_like(masked)
            bias[mid_food] = preferred_safe[mid_food].float() * 0.25
            masked = masked + bias

        return masked

    @torch.no_grad()
    def act(self, obs: np.ndarray, aux: np.ndarray | None = None, deterministic: bool = False):
        x = torch.from_numpy(obs).to(self.device, dtype=torch.float32)
        x = x.permute(0, 3, 1, 2).contiguous()
        aux_t = None
        if aux is not None:
            aux_t = torch.from_numpy(aux).to(self.device, dtype=torch.float32)
        logits, value = self.net(x, aux_t)
        logits = self._apply_action_mask(logits, aux_t)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        logprob = dist.log_prob(action)
        return action.cpu().numpy(), logprob.cpu().numpy(), value.cpu().numpy()

    @torch.no_grad()
    def value(self, obs: np.ndarray, aux: np.ndarray | None = None) -> np.ndarray:
        x = torch.from_numpy(obs).to(self.device, dtype=torch.float32)
        x = x.permute(0, 3, 1, 2).contiguous()
        aux_t = None
        if aux is not None:
            aux_t = torch.from_numpy(aux).to(self.device, dtype=torch.float32)
        _, v = self.net(x, aux_t)
        return v.cpu().numpy()

    def update(self, batch: dict[str, np.ndarray], ent_coef_override: float | None = None) -> dict[str, float]:
        obs = torch.from_numpy(batch["obs"]).to(self.device, dtype=torch.float32)
        obs = obs.permute(0, 3, 1, 2).contiguous()
        aux = torch.from_numpy(batch["aux"]).to(self.device, dtype=torch.float32)
        actions = torch.from_numpy(batch["actions"]).to(self.device, dtype=torch.long)
        old_logprobs = torch.from_numpy(batch["logprobs"]).to(self.device, dtype=torch.float32)
        advantages = torch.from_numpy(batch["advantages"]).to(self.device, dtype=torch.float32)
        returns = torch.from_numpy(batch["returns"]).to(self.device, dtype=torch.float32)
        old_values = torch.from_numpy(batch["values"]).to(self.device, dtype=torch.float32)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = obs.shape[0]
        mb = self.cfg.minibatch_size
        losses_pi: list[float] = []
        losses_v: list[float] = []
        entropies: list[float] = []
        clip_fracs: list[float] = []

        ent_coef = self.cfg.ent_coef if ent_coef_override is None else float(ent_coef_override)

        for _ in range(self.cfg.update_epochs):
            idx = torch.randperm(n, device=self.device)
            for start in range(0, n, mb):
                j = idx[start : start + mb]

                logits, values = self.net(obs[j], aux[j])
                logits = self._apply_action_mask(logits, aux[j])
                dist = Categorical(logits=logits)
                new_logprob = dist.log_prob(actions[j])
                entropy = dist.entropy().mean()

                logratio = new_logprob - old_logprobs[j]
                ratio = logratio.exp()

                pg1 = -advantages[j] * ratio
                pg2 = -advantages[j] * torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef)
                loss_pi = torch.max(pg1, pg2).mean()

                v_pred = values
                v_clipped = old_values[j] + torch.clamp(v_pred - old_values[j], -self.cfg.clip_coef, self.cfg.clip_coef)
                v_loss_1 = F.mse_loss(v_pred, returns[j], reduction="none")
                v_loss_2 = F.mse_loss(v_clipped, returns[j], reduction="none")
                loss_v = 0.5 * torch.max(v_loss_1, v_loss_2).mean()

                loss = loss_pi + self.cfg.vf_coef * loss_v - ent_coef * entropy

                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.optim.step()

                losses_pi.append(float(loss_pi.item()))
                losses_v.append(float(loss_v.item()))
                entropies.append(float(entropy.item()))
                clip_fracs.append(float(((ratio - 1.0).abs() > self.cfg.clip_coef).float().mean().item()))

        self.update_count += 1
        return {
            "loss_pi": float(np.mean(losses_pi)) if losses_pi else 0.0,
            "loss_v": float(np.mean(losses_v)) if losses_v else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "clipfrac": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
        }

    def save(self, path: str, extra: dict | None = None) -> None:
        torch.save(
            {
                "model": self.net.state_dict(),
                "optim": self.optim.state_dict(),
                "cfg": self.cfg.__dict__,
                "update_count": self.update_count,
                "extra": extra or {},
            },
            path,
        )

    def load(self, path: str, strict: bool = True) -> dict:
        payload = torch.load(path, map_location=self.device)
        if "model" in payload:
            self.net.load_state_dict(payload["model"], strict=strict)
            if "optim" in payload:
                self.optim.load_state_dict(payload["optim"])
            self.update_count = int(payload.get("update_count", 0))
            return payload.get("extra", {})

        self.net.load_state_dict(payload, strict=strict)
        return {}
