"""
mappo.py — Multi-Agent PPO trainer.

Implements parameter-sharing MAPPO:
  - One network per team (hiders share, seekers share)
  - Centralized value function: each agent's value head sees
    the concatenated observations of all teammates (CTDE)
  - Rollouts collected across all agents, then minibatch SGD

Key hyperparameters and their intuition:
  clip_eps   : PPO clip range — how far the new policy can drift per update
  gamma      : discount factor — how much future rewards matter
  gae_lambda : GAE smoothing — bias-variance tradeoff in advantage estimate
  ent_coef   : entropy bonus — encourages exploration, prevents collapse
  vf_coef    : value loss weight relative to policy loss
  max_grad   : gradient clipping — stabilises LSTM training
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from env.hide_seek_env import HideSeekEnv, HIDER_IDS, SEEKER_IDS, N_AGENTS
from rl.network import ActorCritic, make_networks, count_params

# ── Hyperparameters ───────────────────────────────────────────────────────────
GAMMA      = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS   = 0.2
ENT_COEF   = 0.01
VF_COEF    = 0.5
MAX_GRAD   = 0.5
LR         = 3e-4
N_EPOCHS   = 4          # PPO update epochs per rollout
BATCH_SIZE = 64         # minibatch size for SGD
ROLLOUT_LEN= 128        # steps collected per rollout


class RolloutBuffer:
    """
    Stores one rollout for a single agent.
    All tensors: (T,) or (T, dim).
    """
    def __init__(self):
        self.obs:       List[np.ndarray] = []
        self.actions:   List[int]        = []
        self.log_probs: List[float]      = []
        self.values:    List[float]      = []
        self.rewards:   List[float]      = []
        self.dones:     List[bool]       = []

    def add(self, obs, action, log_prob, value, reward, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


def compute_gae(rewards: List[float], values: List[float],
                dones: List[bool], last_value: float,
                gamma: float = GAMMA,
                lam: float   = GAE_LAMBDA) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generalised Advantage Estimation.
    Returns (advantages, returns) as float32 arrays.
    """
    T         = len(rewards)
    advs      = np.zeros(T, dtype=np.float32)
    gae       = 0.0
    next_val  = last_value

    for t in reversed(range(T)):
        mask     = 1.0 - float(dones[t])
        delta    = rewards[t] + gamma * next_val * mask - values[t]
        gae      = delta + gamma * lam * mask * gae
        advs[t]  = gae
        next_val = values[t]

    returns = advs + np.array(values, dtype=np.float32)
    return advs, returns


class MAPPOTrainer:
    """
    Trains hiders and seekers simultaneously using parameter-sharing MAPPO.

    Parameters
    ----------
    env        : HideSeekEnv instance
    device     : 'cpu' or 'cuda'
    render     : if True, calls renderer every `render_every` rollouts
    """

    def __init__(self, env: HideSeekEnv, device: str = "cpu",
                 render: bool = False, render_every: int = 10):
        self.env          = env
        self.device       = torch.device(device)
        self.render       = render
        self.render_every = render_every

        self.hider_net, self.seeker_net = make_networks(device)

        self.hider_opt  = torch.optim.Adam(self.hider_net.parameters(),  lr=LR, eps=1e-5)
        self.seeker_opt = torch.optim.Adam(self.seeker_net.parameters(), lr=LR, eps=1e-5)

        # Per-agent buffers
        self.buffers: Dict[int, RolloutBuffer] = {i: RolloutBuffer() for i in range(N_AGENTS)}

        # LSTM hidden states per agent (maintained across rollout, reset per episode)
        self.hidden: Dict[int, Optional[Tuple]] = {i: None for i in range(N_AGENTS)}

        # Stats
        self.total_steps     = 0
        self.episode_returns : Dict[str, List[float]] = defaultdict(list)
        self.losses          : List[Dict] = []

        print(f"Hider  network params : {count_params(self.hider_net):,}")
        print(f"Seeker network params : {count_params(self.seeker_net):,}")
        print(f"Device: {self.device}")

    def _net(self, agent_id: int) -> ActorCritic:
        return self.hider_net if agent_id in HIDER_IDS else self.seeker_net

    def _reset_hidden(self, agent_id: int) -> None:
        self.hidden[agent_id] = None

    def _reset_all_hidden(self) -> None:
        for i in range(N_AGENTS):
            self.hidden[i] = None

    @torch.no_grad()
    def _select_action(self, agent_id: int,
                       obs: np.ndarray) -> Tuple[int, float, float]:
        """Returns (action, log_prob, value)."""
        net = self._net(agent_id)
        obs_t = torch.tensor(obs, dtype=torch.float32,
                             device=self.device).unsqueeze(0)  # (1, OBS_DIM)

        h = self.hidden[agent_id]
        action, log_prob, _, value, new_h = net.get_action_and_value(obs_t, h)
        self.hidden[agent_id] = new_h

        return (action.item(), log_prob.item(), value.item())

    def collect_rollout(self, renderer=None, rollout_idx: int = 0) -> Dict:
        """
        Collect ROLLOUT_LEN steps of experience across all agents.
        Returns episode stats if any episodes completed.
        """
        obs_dict, _ = self.env.reset()
        self._reset_all_hidden()

        ep_rewards = defaultdict(float)
        completed_episodes = 0
        steps_this_rollout = 0

        for buf in self.buffers.values():
            buf.clear()

        for _ in range(ROLLOUT_LEN):
            action_dict = {}
            info_step   = {}   # (action, log_prob, value) per agent

            for agent in self.env.agents:
                aid = agent.agent_id
                if not agent.alive:
                    action_dict[aid] = 0
                    info_step[aid]   = (0, 0.0, 0.0)
                    continue
                obs = obs_dict[aid]
                act, lp, val = self._select_action(aid, obs)
                action_dict[aid] = act
                info_step[aid]   = (act, lp, val)

            next_obs, rewards, dones, _ = self.env.step(action_dict)
            self.total_steps += 1
            steps_this_rollout += 1

            for agent in self.env.agents:
                aid = agent.agent_id
                act, lp, val = info_step[aid]
                self.buffers[aid].add(
                    obs    = obs_dict[aid],
                    action = act,
                    log_prob=lp,
                    value  = val,
                    reward = rewards[aid],
                    done   = dones[aid],
                )
                ep_rewards[aid] += rewards[aid]

            obs_dict = next_obs

            # Render
            if renderer and (rollout_idx % self.render_every == 0):
                alive = renderer.draw(self.env.get_render_state(), fps=20)
                if not alive:
                    return {'render_closed': True}

            if dones['__all__']:
                # Log episode returns
                h_ret = np.mean([ep_rewards[i] for i in HIDER_IDS])
                s_ret = np.mean([ep_rewards[i] for i in SEEKER_IDS])
                self.episode_returns['hider'].append(h_ret)
                self.episode_returns['seeker'].append(s_ret)
                completed_episodes += 1
                ep_rewards = defaultdict(float)

                obs_dict, _ = self.env.reset()
                self._reset_all_hidden()

        return {
            'completed_episodes': completed_episodes,
            'steps': steps_this_rollout,
        }

    def _update_team(self, agent_ids: List[int],
                     net: ActorCritic,
                     opt: torch.optim.Optimizer) -> Dict[str, float]:
        """Run PPO update on collected rollouts for one team."""
        # Concatenate all agents' data
        all_obs, all_acts, all_lps, all_advs, all_rets = [], [], [], [], []

        for aid in agent_ids:
            buf = self.buffers[aid]
            if len(buf) == 0:
                continue

            # Get last value for GAE bootstrap
            with torch.no_grad():
                last_obs = torch.tensor(buf.obs[-1], dtype=torch.float32,
                                        device=self.device).unsqueeze(0)
                _, last_val, _ = net(last_obs, None)
                last_val = last_val.item()

            advs, rets = compute_gae(buf.rewards, buf.values,
                                     buf.dones, last_val)

            all_obs.append(np.array(buf.obs))
            all_acts.append(np.array(buf.actions))
            all_lps.append(np.array(buf.log_probs))
            all_advs.append(advs)
            all_rets.append(rets)

        if not all_obs:
            return {}

        obs_t   = torch.tensor(np.concatenate(all_obs),  dtype=torch.float32, device=self.device)
        acts_t  = torch.tensor(np.concatenate(all_acts), dtype=torch.long,    device=self.device)
        old_lp  = torch.tensor(np.concatenate(all_lps),  dtype=torch.float32, device=self.device)
        advs_t  = torch.tensor(np.concatenate(all_advs), dtype=torch.float32, device=self.device)
        rets_t  = torch.tensor(np.concatenate(all_rets), dtype=torch.float32, device=self.device)

        # Normalise advantages
        advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)

        T = obs_t.shape[0]
        total_pg_loss, total_vf_loss, total_ent = 0.0, 0.0, 0.0
        n_updates = 0

        for _ in range(N_EPOCHS):
            idx = torch.randperm(T, device=self.device)
            for start in range(0, T, BATCH_SIZE):
                mb_idx  = idx[start:start + BATCH_SIZE]
                mb_obs  = obs_t[mb_idx]
                mb_acts = acts_t[mb_idx]
                mb_olp  = old_lp[mb_idx]
                mb_advs = advs_t[mb_idx]
                mb_rets = rets_t[mb_idx]

                # Re-evaluate actions with current net (no LSTM state — truncated BPTT)
                _, new_lp, entropy, new_val, _ = net.get_action_and_value(
                    mb_obs, lstm_state=None, action=mb_acts)

                ratio    = torch.exp(new_lp - mb_olp)
                pg_loss1 = -mb_advs * ratio
                pg_loss2 = -mb_advs * ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                vf_loss  = 0.5 * (new_val - mb_rets).pow(2).mean()
                ent_loss = -entropy.mean()

                loss = pg_loss + VF_COEF * vf_loss + ENT_COEF * ent_loss

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD)
                opt.step()

                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_ent     += (-ent_loss).item()
                n_updates     += 1

        n = max(1, n_updates)
        return {
            'pg_loss': total_pg_loss / n,
            'vf_loss': total_vf_loss / n,
            'entropy': total_ent     / n,
        }

    def train(self, n_rollouts: int = 500,
              save_every: int = 50,
              save_path: str  = "checkpoints",
              renderer=None) -> None:
        """
        Main training loop.

        n_rollouts : number of rollout-update cycles
        save_every : checkpoint interval
        renderer   : HideSeekRenderer instance (optional)
        """
        import os
        os.makedirs(save_path, exist_ok=True)

        for rollout_idx in range(n_rollouts):
            # Collect
            stats = self.collect_rollout(renderer=renderer, rollout_idx=rollout_idx)
            if stats.get('render_closed'):
                print("Renderer closed — stopping training.")
                break

            # Update
            h_loss = self._update_team(HIDER_IDS,  self.hider_net,  self.hider_opt)
            s_loss = self._update_team(SEEKER_IDS, self.seeker_net, self.seeker_opt)

            if self.episode_returns['hider']:
                h_ret = np.mean(self.episode_returns['hider'][-10:])
                s_ret = np.mean(self.episode_returns['seeker'][-10:])
            else:
                h_ret = s_ret = 0.0

            if (rollout_idx + 1) % 10 == 0:
                print(f"[{rollout_idx+1:>4}/{n_rollouts}] "
                      f"steps={self.total_steps:>7,}  "
                      f"H_ret={h_ret:+.1f}  S_ret={s_ret:+.1f}  "
                      f"H_pg={h_loss.get('pg_loss',0):.3f}  "
                      f"H_ent={h_loss.get('entropy',0):.3f}  "
                      f"S_pg={s_loss.get('pg_loss',0):.3f}  "
                      f"S_ent={s_loss.get('entropy',0):.3f}")

            if (rollout_idx + 1) % save_every == 0:
                self.save(f"{save_path}/checkpoint_{rollout_idx+1}.pt")

        self.save(f"{save_path}/final.pt")
        print("Training complete.")

    def save(self, path: str) -> None:
        torch.save({
            'hider_net':  self.hider_net.state_dict(),
            'seeker_net': self.seeker_net.state_dict(),
            'hider_opt':  self.hider_opt.state_dict(),
            'seeker_opt': self.seeker_opt.state_dict(),
            'total_steps':self.total_steps,
            'returns':    dict(self.episode_returns),
        }, path)
        print(f"  Saved → {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.hider_net.load_state_dict(ckpt['hider_net'])
        self.seeker_net.load_state_dict(ckpt['seeker_net'])
        self.hider_opt.load_state_dict(ckpt['hider_opt'])
        self.seeker_opt.load_state_dict(ckpt['seeker_opt'])
        self.total_steps = ckpt.get('total_steps', 0)
        if 'returns' in ckpt:
            self.episode_returns.update(ckpt['returns'])
        print(f"  Loaded ← {path}")