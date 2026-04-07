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
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import copy
from pathlib import Path
from datetime import datetime

from env.hide_seek_env import HideSeekEnv, HIDER_IDS, SEEKER_IDS, N_AGENTS
from rl.network import ActorCritic, make_networks, count_params
from rl.memory import TeamRolloutMemory, to_torch
from rl.checkpointing import safe_torch_load, validate_checkpoint_schema
from rl.monitoring import (
    CSVMetricLogger,
    TensorboardLogger,
    make_run_id,
    prepare_artifacts,
    save_replay_json,
)
from render.video_utils import render_state_to_frame, write_mp4

# ── Hyperparameters ───────────────────────────────────────────────────────────
GAMMA      = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS   = 0.2
ENT_COEF   = 0.01
ENT_COEF_MAX = 0.08
ENTROPY_FLOOR = 0.02
SUCCESS_HIGH = 0.95
VF_COEF    = 0.5
BC_COEF_SEEKER = 0.20
MAX_GRAD   = 0.5
LR         = 3e-4
N_EPOCHS   = 4          # PPO update epochs per rollout
ROLLOUT_LEN= 128        # steps collected per rollout
CKPT_REQUIRED_KEYS = {'hider_net', 'seeker_net'}
SNAPSHOT_INTERVAL = 25
POOL_MAX = 8
OPPONENT_SNAPSHOT_PROB = 0.5


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


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

        # Reusable frozen behavior networks for opponent policy sampling.
        # Reusing avoids per-rollout model allocation/churn on GPU.
        self.hider_behavior_net, self.seeker_behavior_net = make_networks(device)
        self.hider_behavior_net.eval()
        self.seeker_behavior_net.eval()

        self.hider_opt  = torch.optim.Adam(self.hider_net.parameters(),  lr=LR, eps=1e-5)
        self.seeker_opt = torch.optim.Adam(self.seeker_net.parameters(), lr=LR, eps=1e-5)

        self.buffers = TeamRolloutMemory(agent_ids=list(range(N_AGENTS)))

        # Per-agent recurrent states for rollout collection.
        self.hidden: Dict[int, Optional[Tuple]] = {i: None for i in range(N_AGENTS)}

        # Self-play policy pools (frozen snapshots)
        self.hider_pool: List[Dict[str, torch.Tensor]] = []
        self.seeker_pool: List[Dict[str, torch.Tensor]] = []

        # Stats
        self.total_steps     = 0
        self.episode_returns : Dict[str, List[float]] = defaultdict(list)
        self.losses          : List[Dict] = []
        self._team_ent_coef = {
            'hider': float(ENT_COEF),
            'seeker': float(ENT_COEF),
        }
        self._team_success_rate = {
            'hider': 0.0,
            'seeker': 0.0,
        }

        # Keep rollout collection continuous across PPO updates.
        # This avoids resetting env every `ROLLOUT_LEN` steps (which can make
        # the HUD appear stuck at step 128 and prevents full episodes).
        self._cached_obs: Optional[Dict[int, np.ndarray]] = None

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

    def _value_obs(self) -> np.ndarray:
        """Global centralized critic input."""
        return self.env.build_global_state().astype(np.float32)

    def _sample_behavior_state(self, team: str) -> Optional[Dict[str, torch.Tensor]]:
        pool = self.hider_pool if team == 'hider' else self.seeker_pool
        if not pool:
            return None
        if np.random.rand() > OPPONENT_SNAPSHOT_PROB:
            return None
        return copy.deepcopy(pool[np.random.randint(0, len(pool))])

    @torch.no_grad()
    def _select_action_deterministic(self, net: ActorCritic,
                                     obs: np.ndarray,
                                     value_obs: np.ndarray,
                                     action_mask: np.ndarray,
                                     lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor]]:
        obs_t = torch.tensor(obs, dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        value_obs_t = torch.tensor(value_obs, dtype=torch.float32,
                                   device=self.device).unsqueeze(0)
        action_mask_t = torch.tensor(action_mask, dtype=torch.float32,
                                     device=self.device).unsqueeze(0)
        logits, _, new_state = net(
            obs_t,
            lstm_state,
            value_obs=value_obs_t,
            action_mask=action_mask_t,
        )
        action = int(logits.argmax(dim=-1).item())
        return action, new_state

    @torch.no_grad()
    def _select_action(self, net: ActorCritic,
                       obs: np.ndarray,
                       value_obs: np.ndarray,
                       action_mask: np.ndarray,
                       lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[int, float, float, Tuple[torch.Tensor, torch.Tensor]]:
        """Returns (action, log_prob, value)."""
        obs_t = torch.tensor(obs, dtype=torch.float32,
                             device=self.device).unsqueeze(0)  # (1, OBS_DIM)
        value_obs_t = torch.tensor(value_obs, dtype=torch.float32,
                                   device=self.device).unsqueeze(0)
        action_mask_t = torch.tensor(action_mask, dtype=torch.float32,
                                     device=self.device).unsqueeze(0)
        action, log_prob, _, value, new_state = net.get_action_and_value(
            obs_t,
            lstm_state=lstm_state,
            value_obs=value_obs_t,
            action_mask=action_mask_t,
        )

        return (int(action.item()), float(log_prob.item()), float(value.item()), new_state)

    def collect_rollout(self, renderer=None, rollout_idx: int = 0,
                        train_team: str = 'hider',
                        opponent_snapshot: Optional[Dict[str, torch.Tensor]] = None) -> Dict:
        """
        Collect ROLLOUT_LEN steps of experience across all agents.
        Returns episode stats if any episodes completed.
        """
        if self._cached_obs is None:
            obs_dict, _ = self.env.reset()
            self._reset_all_hidden()
            self._cached_obs = obs_dict
        else:
            obs_dict = self._cached_obs

        if train_team == 'hider':
            hider_behavior = self.hider_net
            seeker_behavior = self.seeker_behavior_net
            if opponent_snapshot is not None:
                seeker_behavior.load_state_dict(opponent_snapshot)
            else:
                seeker_behavior.load_state_dict(self.seeker_net.state_dict())
            seeker_behavior.eval()
        else:
            seeker_behavior = self.seeker_net
            hider_behavior = self.hider_behavior_net
            if opponent_snapshot is not None:
                hider_behavior.load_state_dict(opponent_snapshot)
            else:
                hider_behavior.load_state_dict(self.hider_net.state_dict())
            hider_behavior.eval()

        ep_rewards = defaultdict(float)
        ep_len = 0
        episode_records: List[Dict[str, Any]] = []
        completed_episodes = 0
        steps_this_rollout = 0

        self.buffers.clear()

        for _ in range(ROLLOUT_LEN):
            action_dict = {}
            info_step   = {}   # (action, log_prob, value, value_obs, action_mask, pad) per agent
            value_obs = self._value_obs()
            action_masks = self.env.get_action_masks()

            for agent in self.env.agents:
                aid = agent.agent_id
                action_mask = action_masks[aid]
                net = hider_behavior if aid in HIDER_IDS else seeker_behavior

                if not agent.alive:
                    action_dict[aid] = 0
                    info_step[aid]   = (0, 0.0, 0.0, value_obs, action_mask, True)
                    self.hidden[aid] = None
                    continue

                obs = obs_dict[aid]
                act, lp, val, new_state = self._select_action(
                    net=net,
                    obs=obs,
                    value_obs=value_obs,
                    action_mask=action_mask,
                    lstm_state=self.hidden[aid],
                )
                self.hidden[aid] = new_state
                action_dict[aid] = act
                info_step[aid]   = (act, lp, val, value_obs, action_mask, False)

            next_obs, rewards, dones, info_dict = self.env.step(action_dict)
            self.total_steps += 1
            steps_this_rollout += 1
            ep_len += 1

            for agent in self.env.agents:
                aid = agent.agent_id
                if aid in info_step:
                    act, lp, val, value_obs, action_mask, pad = info_step[aid]
                    obs_store = obs_dict[aid]
                else:
                    act, lp, val = (0, 0.0, 0.0)
                    value_obs = self._value_obs()
                    action_mask = self.env.get_action_mask(aid)
                    pad = True
                    obs_store = np.zeros_like(obs_dict[aid], dtype=np.float32)
                self.buffers.add(
                    agent_id = aid,
                    obs = obs_store,
                    action = act,
                    log_prob = lp,
                    value = val,
                    reward = rewards[aid],
                    done = bool(dones[aid]),
                    alive = bool(agent.alive),
                    value_obs = value_obs,
                    action_mask = action_mask,
                    pad = pad,
                    ghost_action = int(info_dict.get(aid, {}).get('ghost_action', 0)),
                    ghost_valid = bool(info_dict.get(aid, {}).get('ghost_valid', False)),
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
                self.episode_returns['hider'].append(float(h_ret))
                self.episode_returns['seeker'].append(float(s_ret))
                completed_episodes += 1
                episode_records.append({
                    'episode_length': int(ep_len),
                    'ep_h_return': float(h_ret),
                    'ep_s_return': float(s_ret),
                    'hiders_caught': int(self.env.hiders_caught),
                    'agent_returns': {int(i): float(ep_rewards[i]) for i in range(N_AGENTS)},
                })
                ep_rewards = defaultdict(float)
                ep_len = 0

                obs_dict, _ = self.env.reset()
                self._reset_all_hidden()

            # Persist env trajectory across rollout boundaries.
            self._cached_obs = obs_dict

        return {
            'completed_episodes': completed_episodes,
            'steps': steps_this_rollout,
            'episodes': episode_records,
        }

    def _update_team(self, agent_ids: List[int],
                     net: ActorCritic,
                     opt: torch.optim.Optimizer,
                     ent_coef: float = ENT_COEF,
                     use_ghost_bc: bool = False) -> Dict[str, float]:
        """Run PPO update on collected rollouts for one team."""
        team_arrs = [self.buffers.arrays_for(aid) for aid in agent_ids]
        if not team_arrs or team_arrs[0]['obs'].shape[0] == 0:
            return {}

        T = team_arrs[0]['obs'].shape[0]
        B = len(team_arrs)

        obs_seq = np.stack([a['obs'] for a in team_arrs], axis=0)
        vobs_seq = np.stack([a['value_obs'] for a in team_arrs], axis=0)
        amask_seq = np.stack([a['action_mask'] for a in team_arrs], axis=0)
        acts_seq = np.stack([a['actions'] for a in team_arrs], axis=0)
        old_lp_seq = np.stack([a['log_probs'] for a in team_arrs], axis=0)
        rewards_seq = np.stack([a['rewards'] for a in team_arrs], axis=0)
        values_seq = np.stack([a['values'] for a in team_arrs], axis=0)
        dones_seq = np.stack([a['dones'] for a in team_arrs], axis=0)
        alive_seq = np.stack([a['alive'] for a in team_arrs], axis=0)
        pad_seq = np.stack([a['pad'] for a in team_arrs], axis=0)
        ghost_actions_seq = np.stack([a['ghost_actions'] for a in team_arrs], axis=0)
        ghost_valid_seq = np.stack([a['ghost_valid'] for a in team_arrs], axis=0)

        advs_seq = np.zeros((B, T), dtype=np.float32)
        rets_seq = np.zeros((B, T), dtype=np.float32)

        tail_values = np.zeros((B,), dtype=np.float32)
        if self._cached_obs is not None:
            value_obs = self._value_obs()
            action_masks = self.env.get_action_masks()
            for i, aid in enumerate(agent_ids):
                # Terminal tail: no bootstrap.
                if bool(dones_seq[i, -1]):
                    tail_values[i] = 0.0
                    continue
                if aid not in self._cached_obs:
                    tail_values[i] = 0.0
                    continue
                obs_t = torch.tensor(self._cached_obs[aid], dtype=torch.float32, device=self.device).unsqueeze(0)
                vobs_t = torch.tensor(value_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                amask_t = torch.tensor(action_masks[aid], dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    _, val_t, _ = net(obs_t, None, value_obs=vobs_t, action_mask=amask_t)
                tail_values[i] = float(val_t.item())

        for i in range(B):
            adv_i, ret_i = compute_gae(
                rewards_seq[i].tolist(),
                values_seq[i].tolist(),
                dones_seq[i].tolist(),
                float(tail_values[i]),
            )
            advs_seq[i] = adv_i
            rets_seq[i] = ret_i

        valid_seq = np.logical_and(alive_seq, np.logical_not(pad_seq))
        if not np.any(valid_seq):
            return {}

        adv_valid = advs_seq[valid_seq]
        adv_mean = float(adv_valid.mean())
        adv_std = float(adv_valid.std()) + 1e-8
        advs_seq = (advs_seq - adv_mean) / adv_std

        obs_t = to_torch(obs_seq, self.device, torch.float32)
        vobs_t = to_torch(vobs_seq, self.device, torch.float32)
        amask_t = to_torch(amask_seq, self.device, torch.float32)
        acts_t = to_torch(acts_seq, self.device, torch.long)
        old_lp_t = to_torch(old_lp_seq, self.device, torch.float32)
        advs_t = to_torch(advs_seq, self.device, torch.float32)
        rets_t = to_torch(rets_seq, self.device, torch.float32)
        valid_t = to_torch(valid_seq.astype(np.float32), self.device, torch.float32)
        ghost_actions_t = to_torch(ghost_actions_seq, self.device, torch.long)
        ghost_valid_t = to_torch(ghost_valid_seq.astype(np.float32), self.device, torch.float32)

        reset_seq = np.zeros((B, T), dtype=np.bool_)
        reset_seq[:, 0] = True
        if T > 1:
            reset_seq[:, 1:] = dones_seq[:, :-1]
        reset_t = to_torch(reset_seq.astype(np.float32), self.device, torch.float32) > 0.5

        total_pg_loss, total_vf_loss, total_ent, total_bc = 0.0, 0.0, 0.0, 0.0
        n_updates = 0
        for _ in range(N_EPOCHS):
            logits_seq, new_val_seq, _ = net.forward_sequence(
                obs_seq=obs_t,
                reset_seq=reset_t,
                value_obs_seq=vobs_t,
                action_mask_seq=amask_t,
            )
            dist = torch.distributions.Categorical(logits=logits_seq)
            new_lp_seq = dist.log_prob(acts_t)
            ent_seq = dist.entropy()

            ratio = torch.exp(new_lp_seq - old_lp_t)
            pg_loss1 = -advs_t * ratio
            pg_loss2 = -advs_t * ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS)
            pg_elem = torch.max(pg_loss1, pg_loss2) * valid_t
            vf_elem = 0.5 * (new_val_seq - rets_t).pow(2) * valid_t
            ent_elem = ent_seq * valid_t

            denom = valid_t.sum().clamp(min=1.0)
            pg_loss = pg_elem.sum() / denom
            vf_loss = vf_elem.sum() / denom
            ent_loss = -(ent_elem.sum() / denom)
            bc_loss = torch.tensor(0.0, device=self.device)
            if use_ghost_bc:
                bc_mask = ghost_valid_t * valid_t
                bc_denom = bc_mask.sum().clamp(min=1.0)
                ce = nn.functional.cross_entropy(
                    logits_seq.reshape(-1, logits_seq.shape[-1]),
                    ghost_actions_t.reshape(-1),
                    reduction='none',
                ).reshape_as(valid_t)
                bc_loss = (ce * bc_mask).sum() / bc_denom

            loss = pg_loss + VF_COEF * vf_loss + ent_coef * ent_loss + BC_COEF_SEEKER * bc_loss
            if not torch.isfinite(loss):
                continue

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD)
            opt.step()

            total_pg_loss += float(pg_loss.item())
            total_vf_loss += float(vf_loss.item())
            total_ent += float((-ent_loss).item())
            total_bc += float(bc_loss.item())
            n_updates += 1

        n = max(1, n_updates)
        return {
            'pg_loss': total_pg_loss / n,
            'vf_loss': total_vf_loss / n,
            'entropy': total_ent     / n,
            'bc_loss': total_bc      / n,
        }

    def _maybe_bump_entropy(self, team: str, latest_entropy: float) -> None:
        coef = float(self._team_ent_coef.get(team, ENT_COEF))
        success = float(self._team_success_rate.get(team, 0.0))
        if latest_entropy <= ENTROPY_FLOOR and success < SUCCESS_HIGH:
            coef = min(ENT_COEF_MAX, coef * 1.35 + 0.002)
        else:
            coef = max(ENT_COEF, coef * 0.995)
        self._team_ent_coef[team] = coef

    def train(self, n_rollouts: int = 500,
              save_every: int = 50,
              save_path: str  = "checkpoints",
              renderer=None,
              output_root: str = "outputs",
              run_id: Optional[str] = None,
              eval_every: int = 0,
              eval_episodes: int = 1,
              eval_fps: int = 12,
              save_eval_videos: bool = True,
              save_replays: bool = True,
              tensorboard: bool = False,
              curriculum_manager: Optional[Any] = None) -> None:
        """
        Main training loop.

        n_rollouts : number of rollout-update cycles
        save_every : checkpoint interval
        renderer   : HideSeekRenderer instance (optional)
        """
        import os
        os.makedirs(save_path, exist_ok=True)

        run_id = run_id or make_run_id("train")
        artifacts = prepare_artifacts(output_root=output_root, run_id=run_id)
        metrics_logger = CSVMetricLogger(artifacts.metrics_csv_path)
        tb_logger = TensorboardLogger(artifacts.tensorboard_dir) if tensorboard else None
        print(f"Run ID: {run_id}")
        print(f"Metrics: {artifacts.metrics_csv_path}")

        # seed pools with initial policies
        self.hider_pool = [copy.deepcopy(self.hider_net.state_dict())]
        self.seeker_pool = [copy.deepcopy(self.seeker_net.state_dict())]

        # Start fresh collection stream for each train() invocation.
        self._cached_obs = None

        try:
            for rollout_idx in range(n_rollouts):
                if curriculum_manager is not None:
                    try:
                        curriculum_manager.on_rollout_start(self, rollout_idx)
                    except Exception as e:
                        print(f"[curriculum] start hook error: {type(e).__name__}: {e}")

                train_team = 'hider' if rollout_idx % 2 == 0 else 'seeker'
                opponent_snapshot = self._sample_behavior_state('seeker' if train_team == 'hider' else 'hider')

                # Collect
                stats = self.collect_rollout(
                    renderer=renderer,
                    rollout_idx=rollout_idx,
                    train_team=train_team,
                    opponent_snapshot=opponent_snapshot,
                )
                if stats.get('render_closed'):
                    print("Renderer closed — stopping training.")
                    break

                # Update
                if train_team == 'hider':
                    h_loss = self._update_team(
                        HIDER_IDS,
                        self.hider_net,
                        self.hider_opt,
                        ent_coef=float(self._team_ent_coef['hider']),
                        use_ghost_bc=False,
                    )
                    s_loss = {}
                else:
                    h_loss = {}
                    s_loss = self._update_team(
                        SEEKER_IDS,
                        self.seeker_net,
                        self.seeker_opt,
                        ent_coef=float(self._team_ent_coef['seeker']),
                        use_ghost_bc=True,
                    )

                eps = stats.get('episodes', []) or []
                if eps:
                    seeker_success = float(np.mean([1.0 if int(ep.get('hiders_caught', 0)) >= 3 else 0.0 for ep in eps]))
                    self._team_success_rate['seeker'] = seeker_success
                    self._team_success_rate['hider'] = 1.0 - seeker_success

                self._maybe_bump_entropy('hider', float(h_loss.get('entropy', 0.0)) if h_loss else 0.0)
                self._maybe_bump_entropy('seeker', float(s_loss.get('entropy', 0.0)) if s_loss else 0.0)

                # B1 fix: reset hidden states after PPO update — old states
                # were produced by pre-update weights and are stale.
                self._reset_all_hidden()

                # snapshot policy pool
                if (rollout_idx + 1) % SNAPSHOT_INTERVAL == 0:
                    self.hider_pool.append(copy.deepcopy(self.hider_net.state_dict()))
                    self.seeker_pool.append(copy.deepcopy(self.seeker_net.state_dict()))
                    self.hider_pool = self.hider_pool[-POOL_MAX:]
                    self.seeker_pool = self.seeker_pool[-POOL_MAX:]

                if self.episode_returns['hider']:
                    h_ret = np.mean(self.episode_returns['hider'][-10:])
                    s_ret = np.mean(self.episode_returns['seeker'][-10:])
                else:
                    h_ret = s_ret = 0.0

                eval_result: Dict[str, Any] = {}
                eval_video_name = ""
                eval_replay_name = ""
                if eval_every > 0 and (rollout_idx + 1) % eval_every == 0:
                    try:
                        eval_result, eval_video_name, eval_replay_name = self._run_evaluation(
                            iteration=rollout_idx + 1,
                            episodes=eval_episodes,
                            fps=eval_fps,
                            video_dir=artifacts.eval_video_dir,
                            replay_dir=artifacts.replay_dir,
                            save_video=save_eval_videos,
                            save_replay=save_replays,
                        )
                    except Exception as e:
                        eval_result = {'error': f'{type(e).__name__}: {e}'}
                        eval_video_name = ""
                        eval_replay_name = ""

                metrics_logger.log({
                    'timestamp': datetime.utcnow().isoformat(),
                    'run_id': run_id,
                    'row_type': 'rollout',
                    'rollout': rollout_idx + 1,
                    'train_team': train_team,
                    'total_steps': self.total_steps,
                    'rollout_steps': stats.get('steps', 0),
                    'completed_episodes_in_rollout': stats.get('completed_episodes', 0),
                    'episode_index': '',
                    'episode_length': '',
                    'ep_h_return': '',
                    'ep_s_return': '',
                    'agent_return_0': '',
                    'agent_return_1': '',
                    'agent_return_2': '',
                    'agent_return_3': '',
                    'agent_return_4': '',
                    'agent_return_5': '',
                    'h_return_last10': _safe_float(h_ret),
                    's_return_last10': _safe_float(s_ret),
                    'lr_h': _safe_float(self.hider_opt.param_groups[0].get('lr', 0.0)),
                    'lr_s': _safe_float(self.seeker_opt.param_groups[0].get('lr', 0.0)),
                    'h_pg_loss': _safe_float(h_loss.get('pg_loss', 0.0)),
                    'h_vf_loss': _safe_float(h_loss.get('vf_loss', 0.0)),
                    'h_entropy': _safe_float(h_loss.get('entropy', 0.0)),
                    's_pg_loss': _safe_float(s_loss.get('pg_loss', 0.0)),
                    's_vf_loss': _safe_float(s_loss.get('vf_loss', 0.0)),
                    's_entropy': _safe_float(s_loss.get('entropy', 0.0)),
                    'eval_h_return': _safe_float(eval_result.get('h_return', 0.0)),
                    'eval_s_return': _safe_float(eval_result.get('s_return', 0.0)),
                    'eval_ep_length': _safe_float(eval_result.get('episode_length', 0.0)),
                    'eval_hiders_caught': _safe_float(eval_result.get('hiders_caught', 0.0)),
                    'eval_video': eval_video_name,
                    'eval_replay': str(eval_result.get('replay_files', eval_replay_name)),
                    'eval_error': str(eval_result.get('error', '')),
                })

                for ep_idx, ep_rec in enumerate(stats.get('episodes', []), start=1):
                    a_ret = ep_rec.get('agent_returns', {})
                    metrics_logger.log({
                        'timestamp': datetime.utcnow().isoformat(),
                        'run_id': run_id,
                        'row_type': 'episode',
                        'rollout': rollout_idx + 1,
                        'train_team': train_team,
                        'total_steps': self.total_steps,
                        'rollout_steps': stats.get('steps', 0),
                        'completed_episodes_in_rollout': stats.get('completed_episodes', 0),
                        'episode_index': ep_idx,
                        'episode_length': ep_rec.get('episode_length', 0),
                        'ep_h_return': ep_rec.get('ep_h_return', 0.0),
                        'ep_s_return': ep_rec.get('ep_s_return', 0.0),
                        'agent_return_0': _safe_float(a_ret.get(0, 0.0)),
                        'agent_return_1': _safe_float(a_ret.get(1, 0.0)),
                        'agent_return_2': _safe_float(a_ret.get(2, 0.0)),
                        'agent_return_3': _safe_float(a_ret.get(3, 0.0)),
                        'agent_return_4': _safe_float(a_ret.get(4, 0.0)),
                        'agent_return_5': _safe_float(a_ret.get(5, 0.0)),
                        'h_return_last10': '',
                        's_return_last10': '',
                        'lr_h': _safe_float(self.hider_opt.param_groups[0].get('lr', 0.0)),
                        'lr_s': _safe_float(self.seeker_opt.param_groups[0].get('lr', 0.0)),
                        'h_pg_loss': '',
                        'h_vf_loss': '',
                        'h_entropy': '',
                        's_pg_loss': '',
                        's_vf_loss': '',
                        's_entropy': '',
                        'eval_h_return': '',
                        'eval_s_return': '',
                        'eval_ep_length': '',
                        'eval_hiders_caught': '',
                        'eval_video': '',
                        'eval_replay': '',
                        'eval_error': '',
                    })

                if tb_logger is not None and tb_logger.enabled:
                    step = rollout_idx + 1
                    tb_logger.scalar('train/return_hider_last10', float(h_ret), step)
                    tb_logger.scalar('train/return_seeker_last10', float(s_ret), step)
                    tb_logger.scalar('train/hider_pg_loss', float(h_loss.get('pg_loss', 0.0)), step)
                    tb_logger.scalar('train/hider_vf_loss', float(h_loss.get('vf_loss', 0.0)), step)
                    tb_logger.scalar('train/hider_entropy', float(h_loss.get('entropy', 0.0)), step)
                    tb_logger.scalar('train/seeker_pg_loss', float(s_loss.get('pg_loss', 0.0)), step)
                    tb_logger.scalar('train/seeker_vf_loss', float(s_loss.get('vf_loss', 0.0)), step)
                    tb_logger.scalar('train/seeker_entropy', float(s_loss.get('entropy', 0.0)), step)
                    if eval_result:
                        tb_logger.scalar('eval/h_return', float(eval_result.get('h_return', 0.0)), step)
                        tb_logger.scalar('eval/s_return', float(eval_result.get('s_return', 0.0)), step)
                        tb_logger.scalar('eval/episode_length', float(eval_result.get('episode_length', 0.0)), step)
                        tb_logger.scalar('eval/hiders_caught', float(eval_result.get('hiders_caught', 0.0)), step)

                if (rollout_idx + 1) % 10 == 0:
                    print(f"[{rollout_idx+1:>4}/{n_rollouts}] "
                          f"team={train_team}  "
                          f"steps={self.total_steps:>7,}  "
                          f"H_ret={h_ret:+.1f}  S_ret={s_ret:+.1f}  "
                          f"H_pg={h_loss.get('pg_loss',0):.3f}  "
                          f"H_ent={h_loss.get('entropy',0):.3f}  "
                          f"H_ec={self._team_ent_coef.get('hider', ENT_COEF):.3f}  "
                          f"S_pg={s_loss.get('pg_loss',0):.3f}  "
                          f"S_ent={s_loss.get('entropy',0):.3f}  "
                          f"S_ec={self._team_ent_coef.get('seeker', ENT_COEF):.3f}  "
                          f"S_bc={s_loss.get('bc_loss',0):.3f}")

                if (rollout_idx + 1) % save_every == 0:
                    self.save(f"{save_path}/checkpoint_{rollout_idx+1}.pt")

                if curriculum_manager is not None:
                    try:
                        curriculum_manager.on_rollout_end(self, rollout_idx, stats)
                    except Exception as e:
                        print(f"[curriculum] end hook error: {type(e).__name__}: {e}")

        finally:
            metrics_logger.close()
            if tb_logger is not None:
                tb_logger.close()

        self.save(f"{save_path}/final.pt")
        print("Training complete.")

    @torch.no_grad()
    def _run_evaluation(self,
                        iteration: int,
                        episodes: int,
                        fps: int,
                        video_dir: Path,
                        replay_dir: Path,
                        save_video: bool,
                        save_replay: bool) -> Tuple[Dict[str, float], str, str]:
        eval_env = HideSeekEnv(
            width=self.env.gen.width,
            height=self.env.gen.height,
            n_food=self.env.gen.n_food,
            n_heavy=self.env.gen.n_heavy_obj,
            max_steps=self.env.max_steps,
            prep_steps=self.env.prep_steps,
            seed=self.env.base_seed,
        )

        h_returns: List[float] = []
        s_returns: List[float] = []
        lengths: List[float] = []
        catches: List[float] = []
        all_frames: List[np.ndarray] = []
        replay_file_name = ""
        replay_names: List[str] = []
        errors: List[str] = []

        for ep in range(max(1, episodes)):
            obs_dict, _ = eval_env.reset(seed=(iteration * 1000 + ep))
            hidden: Dict[int, Optional[Tuple[torch.Tensor, torch.Tensor]]] = {i: None for i in range(N_AGENTS)}
            ep_rewards = {i: 0.0 for i in range(N_AGENTS)}
            ep_states: List[Dict[str, Any]] = []
            ep_len = 0

            while True:
                action_dict: Dict[int, int] = {}
                value_obs = eval_env.build_global_state().astype(np.float32)
                action_masks = eval_env.get_action_masks()

                for agent in eval_env.agents:
                    aid = agent.agent_id
                    if not agent.alive:
                        action_dict[aid] = 0
                        hidden[aid] = None
                        continue

                    net = self.hider_net if aid in HIDER_IDS else self.seeker_net
                    act, new_state = self._select_action_deterministic(
                        net=net,
                        obs=obs_dict[aid],
                        value_obs=value_obs,
                        action_mask=action_masks[aid],
                        lstm_state=hidden[aid],
                    )
                    hidden[aid] = new_state
                    action_dict[aid] = act

                obs_dict, rewards, dones, _ = eval_env.step(action_dict)
                ep_len += 1
                for i, r in rewards.items():
                    ep_rewards[i] += float(r)

                render_state = eval_env.get_render_state()
                if save_video and ep == 0:
                    all_frames.append(render_state_to_frame(render_state, tile_px=10))
                if save_replay:
                    ep_states.append(eval_env.get_serializable_render_state())

                if dones['__all__']:
                    break

            h_returns.append(float(np.mean([ep_rewards[i] for i in HIDER_IDS])))
            s_returns.append(float(np.mean([ep_rewards[i] for i in SEEKER_IDS])))
            lengths.append(float(ep_len))
            catches.append(float(eval_env.hiders_caught))

            if save_replay:
                replay_name = f"iter_{iteration:06d}_ep_{ep+1:02d}.json"
                replay_path = replay_dir / replay_name
                try:
                    save_replay_json(replay_path, {
                        'schema_version': 1,
                        'producer': 'hide_and_seek_agents',
                        'env': {
                            'width': int(eval_env.gen.width),
                            'height': int(eval_env.gen.height),
                            'max_steps': int(eval_env.max_steps),
                            'prep_steps': int(eval_env.prep_steps),
                        },
                        'iteration': iteration,
                        'episode': ep + 1,
                        'frames': ep_states,
                        'summary': {
                            'h_return': h_returns[-1],
                            's_return': s_returns[-1],
                            'episode_length': lengths[-1],
                            'hiders_caught': catches[-1],
                        },
                    })
                    replay_file_name = replay_name
                    replay_names.append(replay_name)
                except Exception as e:
                    errors.append(f'replay_ep_{ep+1}:{type(e).__name__}:{e}')

        video_name = ""
        if save_video and all_frames:
            video_name = f"{iteration:06d}.mp4"
            try:
                write_mp4(video_dir / video_name, all_frames, fps=fps)
            except Exception as e:
                video_name = ""
                errors.append(f'video:{type(e).__name__}:{e}')

        result = {
            'h_return': float(np.mean(h_returns)) if h_returns else 0.0,
            's_return': float(np.mean(s_returns)) if s_returns else 0.0,
            'episode_length': float(np.mean(lengths)) if lengths else 0.0,
            'hiders_caught': float(np.mean(catches)) if catches else 0.0,
            'replay_files': ';'.join(replay_names),
            'error': ' | '.join(errors),
        }
        replay_file_name = ';'.join(replay_names) if replay_names else replay_file_name
        return result, video_name, replay_file_name

    def save(self, path: str) -> None:
        torch.save({
            'hider_net':  self.hider_net.state_dict(),
            'seeker_net': self.seeker_net.state_dict(),
            'hider_opt':  self.hider_opt.state_dict(),
            'seeker_opt': self.seeker_opt.state_dict(),
            'total_steps':self.total_steps,
            'returns':    dict(self.episode_returns),
            'hider_pool': self.hider_pool,
            'seeker_pool': self.seeker_pool,
        }, path)
        print(f"  Saved → {path}")

    def load(self, path: str) -> None:
        ckpt = safe_torch_load(path, map_location=self.device)
        validate_checkpoint_schema(ckpt, CKPT_REQUIRED_KEYS)
        self.hider_net.load_state_dict(ckpt['hider_net'])
        self.seeker_net.load_state_dict(ckpt['seeker_net'])
        if 'hider_opt' in ckpt:
            self.hider_opt.load_state_dict(ckpt['hider_opt'])
        if 'seeker_opt' in ckpt:
            self.seeker_opt.load_state_dict(ckpt['seeker_opt'])
        self.total_steps = ckpt.get('total_steps', 0)
        if 'returns' in ckpt:
            self.episode_returns.update(ckpt['returns'])
        if 'hider_pool' in ckpt and isinstance(ckpt['hider_pool'], list):
            self.hider_pool = ckpt['hider_pool'][-POOL_MAX:]
        if 'seeker_pool' in ckpt and isinstance(ckpt['seeker_pool'], list):
            self.seeker_pool = ckpt['seeker_pool'][-POOL_MAX:]
        print(f"  Loaded ← {path}")