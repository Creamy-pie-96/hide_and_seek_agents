from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import torch

from .env_v2 import SnakeEnv
from .ppo_agent import PPOAgent, PPOConfig


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


@dataclass
class TrainPPOConfig:
    episodes: int = 3000
    grid_size: int = 20
    seed: int = 42
    out_dir: str = "."
    device: str = "auto"

    # PPO
    lr: float = 4e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.006
    ent_coef_end: float = 0.0005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_envs: int = 12
    rollout_steps: int = 256
    update_epochs: int = 4
    minibatch_size: int = 512

    # reward/env shaping
    max_steps_factor: int = 100
    loop_visit_threshold: int = 3
    loop_visit_penalty: float = -0.03
    food_reward: float = 15.0
    death_penalty: float = -8.0
    distance_reward_toward: float = 0.2
    distance_penalty_away: float = -0.1
    stagnation_penalty: float = 0.0
    survival_reward: float = -0.003
    idle_step_coeff: float = 0.0
    hunger_exp_base: float = 0.003
    hunger_exp_gamma: float = 1.02
    hunger_exp_max: float = 1.2
    step_limit_penalty: float = -8.0
    starvation_steps_factor: int = 55
    starvation_penalty: float = -5.0
    wall_follow_threshold: int = 5
    wall_follow_penalty: float = -0.2
    obstacle_count: int = 0
    moving_obstacles: bool = False
    obstacle_move_period: int = 8
    moving_food: bool = False
    food_move_prob: float = 0.15
    train_random_start: bool = True
    eval_random_start: bool = True
    obs_noise_std: float = 0.005

    # training strategy
    use_curriculum: bool = True
    curriculum_promote_streak: int = 3
    curriculum_prev_mix_prob: float = 0.30
    curriculum_history_mix_prob: float = 0.20
    randomize_train_seeds: bool = True
    reseed_every_updates: int = 12
    reseed_span: int = 1_000_000

    # adaptive entropy control
    use_adaptive_entropy: bool = True
    entropy_min: float = 1.5e-3
    entropy_max: float = 2e-2
    entropy_up_step: float = 1.2e-3
    entropy_down_step: float = 4e-4
    entropy_plateau_patience: int = 2
    pro_entropy_floor: float = 0.0018
    use_regression_rollback: bool = False

    # self-play/static opponent
    self_play: bool = False
    self_play_mode: str = "heuristic"  # heuristic | last_best
    opponent_food_penalty: float = -2.0
    opponent_heuristic_random_prob: float = 0.15
    terminal_win_reward: float = 8.0
    terminal_loss_penalty: float = -8.0
    self_play_warmup_episodes: int = 200
    self_play_best_prob: float = 0.7
    self_play_min_eval_score: float = 1.0

    # resume
    resume: str = ""
    resume_reset_optim: bool = False

    # eval/checkpoints
    eval_every: int = 25
    eval_episodes: int = 10
    eval_seed_batches: int = 1
    save_every: int = 100
    log_every: int = 10


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def curriculum_values(progress: float) -> dict:
    p = float(np.clip(progress, 0.0, 1.0))
    return {
        "max_steps_factor": int(round(50 + 50 * p)),
        "distance_reward_toward": float(0.08 - 0.03 * p),
        "distance_penalty_away": float(-0.08 + 0.01 * p),
        "loop_visit_penalty": float(-0.01 - 0.02 * p),
        "step_limit_penalty": float(-1.0),
        "food_spawn_radius": None if p >= 0.65 else int(round(6 - 5 * p)),
    }


class CurriculumManager:
    def __init__(self, promote_streak: int = 3):
        self.promote_streak = max(1, int(promote_streak))
        self.level = 0
        self.streak = 0
        self.fail_streak = 0
        self.eval_ema = 0.0
        self.eval_ema_alpha = 0.30
        self.min_level_episodes = 120
        self.demote_streak = 4
        self.demote_margin = 1.6
        self.level_entry_episode = 0
        self.min_margin_by_level = [-999.0, -999.0, -999.0, -999.0, -1.5]
        self.max_step_limit_rate_by_level = [0.85, 0.75, 0.65, 0.60, 0.55]
        self.min_food_per_100_by_level = [0.10, 0.18, 0.28, 0.40, 0.55]
        self.levels = [
            {
                "name": "solo_static_food",
                "score_threshold": 0.45,
                "params": {
                    "opponent_mode": "none",
                    "obstacle_count": 0,
                    "moving_obstacles": False,
                    "moving_food": False,
                    "max_steps_factor": 55,
                    "starvation_steps_factor": 54,
                    "food_spawn_radius": 6,
                    "loop_visit_penalty": -0.01,
                },
            },
            {
                "name": "solo_static_blocks",
                "score_threshold": 0.65,
                "params": {
                    "opponent_mode": "none",
                    "obstacle_count": 6,
                    "moving_obstacles": False,
                    "moving_food": False,
                    "max_steps_factor": 60,
                    "starvation_steps_factor": 50,
                    "food_spawn_radius": 5,
                    "loop_visit_penalty": -0.015,
                },
            },
            {
                "name": "solo_moving_blocks",
                "score_threshold": 0.80,
                "params": {
                    "opponent_mode": "none",
                    "obstacle_count": 8,
                    "moving_obstacles": True,
                    "obstacle_move_period": 10,
                    "moving_food": False,
                    "max_steps_factor": 70,
                    "starvation_steps_factor": 48,
                    "food_spawn_radius": 3,
                    "loop_visit_penalty": -0.02,
                },
            },
            {
                "name": "solo_moving_blocks_food",
                "score_threshold": 1.00,
                "params": {
                    "opponent_mode": "none",
                    "obstacle_count": 10,
                    "moving_obstacles": True,
                    "obstacle_move_period": 9,
                    "moving_food": True,
                    "food_move_prob": 0.08,
                    "max_steps_factor": 80,
                    "starvation_steps_factor": 46,
                    "food_spawn_radius": 3,
                    "loop_visit_penalty": -0.02,
                },
            },
            {
                "name": "competitive",
                "score_threshold": 1.20,
                "params": {
                    "opponent_mode": "heuristic",
                    "obstacle_count": 10,
                    "moving_obstacles": True,
                    "obstacle_move_period": 7,
                    "moving_food": True,
                    "food_move_prob": 0.10,
                    "max_steps_factor": 100,
                    "starvation_steps_factor": 48,
                    "food_spawn_radius": 5,
                    "loop_visit_penalty": -0.015,
                    "opponent_food_penalty": -0.10,
                    "opponent_random_prob": 0.80,
                },
            },
        ]

    def current_params(self) -> dict:
        return dict(self.levels[self.level]["params"])

    def params_for_level(self, level: int) -> dict:
        idx = int(np.clip(level, 0, len(self.levels) - 1))
        return dict(self.levels[idx]["params"])

    def on_eval(
        self,
        eval_avg_score: float,
        episode_now: int,
        eval_avg_margin: float | None = None,
        eval_step_limit_rate: float | None = None,
        eval_food_per_100_steps: float | None = None,
    ) -> tuple[bool, str]:
        self.eval_ema = self.eval_ema_alpha * float(eval_avg_score) + (1.0 - self.eval_ema_alpha) * self.eval_ema
        threshold = float(self.levels[self.level]["score_threshold"])
        margin_ok = True
        quality_ok = True
        if eval_avg_margin is not None:
            min_margin = float(self.min_margin_by_level[min(self.level, len(self.min_margin_by_level) - 1)])
            margin_ok = float(eval_avg_margin) >= min_margin
        if eval_step_limit_rate is not None:
            max_step_limit = float(self.max_step_limit_rate_by_level[min(self.level, len(self.max_step_limit_rate_by_level) - 1)])
            quality_ok = quality_ok and (float(eval_step_limit_rate) <= max_step_limit)
        if eval_food_per_100_steps is not None:
            min_food_rate = float(self.min_food_per_100_by_level[min(self.level, len(self.min_food_per_100_by_level) - 1)])
            quality_ok = quality_ok and (float(eval_food_per_100_steps) >= min_food_rate)

        if self.eval_ema >= threshold and margin_ok and quality_ok:
            self.fail_streak = 0
        elif self.level > 0 and (self.eval_ema < (threshold - self.demote_margin) or not quality_ok):
            self.fail_streak += 1
        else:
            self.fail_streak = max(0, self.fail_streak - 1)

        if self.level > 0 and self.fail_streak >= self.demote_streak:
            self.level -= 1
            self.streak = 0
            self.fail_streak = 0
            self.level_entry_episode = int(episode_now)
            return True, f"demoted_to_{self.levels[self.level]['name']}"

        if self.level >= len(self.levels) - 1:
            return False, "max_level"

        if int(episode_now) - self.level_entry_episode < self.min_level_episodes:
            return False, "hold_min_level_time"
        if self.eval_ema >= threshold and margin_ok and quality_ok:
            self.streak += 1
        else:
            self.streak = max(0, self.streak - 1)
        if self.streak >= self.promote_streak:
            self.level += 1
            self.streak = 0
            self.fail_streak = 0
            self.level_entry_episode = int(episode_now)
            return True, f"promoted_to_{self.levels[self.level]['name']}"
        return False, "hold"


class AdaptiveEntropyController:
    def __init__(
        self,
        base_start: float,
        base_end: float,
        min_coef: float,
        max_coef: float,
        up_step: float,
        down_step: float,
        plateau_patience: int,
    ):
        self.base_start = float(base_start)
        self.base_end = float(base_end)
        self.min_coef = float(min_coef)
        self.max_coef = float(max_coef)
        self.up_step = float(up_step)
        self.down_step = float(down_step)
        self.plateau_patience = max(1, int(plateau_patience))
        self.current = float(base_start)
        self.best_eval = -1e9
        self.no_improve = 0

    def scheduled(self, progress: float) -> float:
        p = float(np.clip(progress, 0.0, 1.0))
        return self.base_start + p * (self.base_end - self.base_start)

    def get(self, progress: float) -> float:
        base = self.scheduled(progress)
        blended = 0.5 * self.current + 0.5 * base
        self.current = float(np.clip(blended, self.min_coef, self.max_coef))
        return self.current

    def update_from_eval(self, eval_avg_score: float, entropy: float) -> str:
        reason = "keep"
        if eval_avg_score > self.best_eval + 1e-6:
            self.best_eval = eval_avg_score
            self.no_improve = 0
            self.current = float(np.clip(self.current - self.down_step, self.min_coef, self.max_coef))
            reason = "improving_decay"
        else:
            self.no_improve += 1
            if self.no_improve >= self.plateau_patience and entropy < 0.35:
                self.current = float(np.clip(self.current + self.up_step, self.min_coef, self.max_coef))
                self.no_improve = 0
                reason = "plateau_boost"
        return reason


class SyncVecSnake:
    def __init__(self, cfg: TrainPPOConfig) -> None:
        opponent_mode = "none"
        if cfg.self_play:
            opponent_mode = "heuristic" if cfg.self_play_mode == "heuristic" else "frozen"

        self.envs = [
            SnakeEnv(
                grid_size=cfg.grid_size,
                seed=cfg.seed + i,
                max_steps_factor=cfg.max_steps_factor,
                loop_visit_threshold=cfg.loop_visit_threshold,
                loop_visit_penalty=cfg.loop_visit_penalty,
                food_reward=cfg.food_reward,
                death_penalty=cfg.death_penalty,
                distance_reward_toward=cfg.distance_reward_toward,
                distance_penalty_away=cfg.distance_penalty_away,
                stagnation_penalty=cfg.stagnation_penalty,
                survival_reward=cfg.survival_reward,
                idle_step_coeff=cfg.idle_step_coeff,
                hunger_exp_base=cfg.hunger_exp_base,
                hunger_exp_gamma=cfg.hunger_exp_gamma,
                hunger_exp_max=cfg.hunger_exp_max,
                step_limit_penalty=cfg.step_limit_penalty,
                starvation_steps_factor=cfg.starvation_steps_factor,
                starvation_penalty=cfg.starvation_penalty,
                wall_follow_threshold=cfg.wall_follow_threshold,
                wall_follow_penalty=cfg.wall_follow_penalty,
                obstacle_count=cfg.obstacle_count,
                moving_obstacles=cfg.moving_obstacles,
                obstacle_move_period=cfg.obstacle_move_period,
                moving_food=cfg.moving_food,
                food_move_prob=cfg.food_move_prob,
                food_spawn_radius=None,
                random_start=cfg.train_random_start,
                opponent_mode=opponent_mode,
                opponent_food_penalty=cfg.opponent_food_penalty,
                opponent_random_prob=cfg.opponent_heuristic_random_prob,
                terminal_win_reward=cfg.terminal_win_reward,
                terminal_loss_penalty=cfg.terminal_loss_penalty,
            )
            for i in range(cfg.n_envs)
        ]
        self.obs_channels = int(self.envs[0].obs_channels) if self.envs else 4

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        obs = np.stack([e.reset() for e in self.envs], axis=0).astype(np.float32)
        aux = np.stack([e.aux_features() for e in self.envs], axis=0).astype(np.float32)
        return obs, aux

    def step(
        self,
        actions: np.ndarray,
        opponent_agent: PPOAgent | None = None,
        use_best_mask: np.ndarray | None = None,
    ):
        next_obs = []
        next_aux = []
        rewards = np.zeros((len(self.envs),), dtype=np.float32)
        dones = np.zeros((len(self.envs),), dtype=np.float32)
        infos: list[dict] = []

        for i, (e, a) in enumerate(zip(self.envs, actions)):
            opp_action = None
            use_best = bool(use_best_mask[i]) if use_best_mask is not None else True
            if opponent_agent is not None and e.has_opponent and use_best:
                o_obs = np.expand_dims(e.opponent_observation(), axis=0)
                o_aux = np.expand_dims(e.opponent_aux_features(), axis=0)
                oa, _, _ = opponent_agent.act(o_obs, o_aux, deterministic=True)
                opp_action = int(oa[0])

            ns, r, done, trunc, info = e.step(int(a), opponent_action=opp_action)
            terminal = bool(done or trunc)
            rewards[i] = float(r)
            dones[i] = float(terminal)
            infos.append(info)
            if terminal:
                ns = e.reset()
            na = e.aux_features()
            next_obs.append(ns)
            next_aux.append(na)

        return (
            np.stack(next_obs, axis=0).astype(np.float32),
            np.stack(next_aux, axis=0).astype(np.float32),
            rewards,
            dones,
            infos,
        )

    def apply_curriculum_params(self, params: dict) -> None:
        for e in self.envs:
            e.set_curriculum(**params)

    def apply_curriculum_mixture(self, current_params: dict, previous_params: dict, previous_prob: float) -> None:
        p_prev = float(np.clip(previous_prob, 0.0, 1.0))
        for e in self.envs:
            if random.random() < p_prev:
                e.set_curriculum(**previous_params)
            else:
                e.set_curriculum(**current_params)

    def apply_curriculum_params_per_env(self, params_per_env: list[dict]) -> None:
        if len(params_per_env) != len(self.envs):
            raise ValueError("params_per_env length must match number of envs")
        for e, p in zip(self.envs, params_per_env):
            e.set_curriculum(**p)

    def close(self) -> None:
        for e in self.envs:
            e.close()

    def reseed_all(self, base_seed: int, span: int = 1_000_000) -> None:
        s = max(1, int(span))
        b = int(base_seed)
        for i, e in enumerate(self.envs):
            e.reseed(b + (i * 7919) % s)


def compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    next_value: np.ndarray,
    gamma: float,
    gae_lambda: float,
):
    t_steps, n_envs = rewards.shape
    adv = np.zeros((t_steps, n_envs), dtype=np.float32)
    lastgaelam = np.zeros((n_envs,), dtype=np.float32)

    for t in reversed(range(t_steps)):
        if t == t_steps - 1:
            nextnonterminal = 1.0 - dones[t]
            nextvalues = next_value
        else:
            # Non-terminal mask belongs to transition at time t: (s_t, a_t, r_t, s_{t+1})
            nextnonterminal = 1.0 - dones[t]
            nextvalues = values[t + 1]

        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        adv[t] = lastgaelam

    returns = adv + values
    return adv, returns


def evaluate(
    agent: PPOAgent,
    cfg: TrainPPOConfig,
    episodes: int,
    curriculum_params: dict | None = None,
    opponent_agent: PPOAgent | None = None,
    seed_offset: int = 0,
):
    traces = []
    scores: list[int] = []
    rewards_list: list[float] = []
    steps_list: list[int] = []
    margins: list[float] = []
    terminal_reasons: list[str] = []
    foods_per_100: list[float] = []

    for ep in range(episodes):
        env = SnakeEnv(
            grid_size=cfg.grid_size,
            seed=cfg.seed + 999 + int(seed_offset) * 1003 + ep,
            max_steps_factor=cfg.max_steps_factor,
            loop_visit_threshold=cfg.loop_visit_threshold,
            loop_visit_penalty=cfg.loop_visit_penalty,
            food_reward=cfg.food_reward,
            death_penalty=cfg.death_penalty,
            distance_reward_toward=cfg.distance_reward_toward,
            distance_penalty_away=cfg.distance_penalty_away,
            stagnation_penalty=cfg.stagnation_penalty,
            survival_reward=cfg.survival_reward,
            idle_step_coeff=cfg.idle_step_coeff,
            hunger_exp_base=cfg.hunger_exp_base,
            hunger_exp_gamma=cfg.hunger_exp_gamma,
            hunger_exp_max=cfg.hunger_exp_max,
            step_limit_penalty=cfg.step_limit_penalty,
            starvation_steps_factor=cfg.starvation_steps_factor,
            starvation_penalty=cfg.starvation_penalty,
            wall_follow_threshold=cfg.wall_follow_threshold,
            wall_follow_penalty=cfg.wall_follow_penalty,
            obstacle_count=cfg.obstacle_count,
            moving_obstacles=cfg.moving_obstacles,
            obstacle_move_period=cfg.obstacle_move_period,
            moving_food=cfg.moving_food,
            food_move_prob=cfg.food_move_prob,
            food_spawn_radius=None,
            random_start=cfg.eval_random_start,
            opponent_mode=("heuristic" if cfg.self_play and cfg.self_play_mode == "heuristic" else ("frozen" if cfg.self_play else "none")),
            opponent_food_penalty=cfg.opponent_food_penalty,
            opponent_random_prob=cfg.opponent_heuristic_random_prob,
            terminal_win_reward=cfg.terminal_win_reward,
            terminal_loss_penalty=cfg.terminal_loss_penalty,
        )
        if curriculum_params:
            env.set_curriculum(**curriculum_params)

        s = env.reset()
        ep_reward = 0.0
        traj = []
        for t in range(1, 5000):
            aux = env.aux_features()
            a, _, _ = agent.act(np.expand_dims(s, axis=0), np.expand_dims(aux, axis=0), deterministic=True)
            opp_action = None
            if opponent_agent is not None and env.has_opponent:
                o_obs = np.expand_dims(env.opponent_observation(), axis=0)
                o_aux = np.expand_dims(env.opponent_aux_features(), axis=0)
                oa, _, _ = opponent_agent.act(o_obs, o_aux, deterministic=True)
                opp_action = int(oa[0])

            ns, r, done, trunc, info = env.step(int(a[0]), opponent_action=opp_action)
            ep_reward += float(r)
            traj.append(
                {
                    "t": t,
                    "action": int(a[0]),
                    "reward": float(r),
                    "score": int(info["score"]),
                    "opponent_score": int(info.get("opponent_score", 0)),
                    "length": int(info["length"]),
                    "reason": info["reason"],
                }
            )
            s = ns
            if done or trunc:
                scores.append(int(info["score"]))
                margins.append(float(int(info["score"]) - int(info.get("opponent_score", 0))))
                rewards_list.append(float(ep_reward))
                steps_list.append(int(t))
                terminal_reasons.append(str(info.get("reason", "unknown")))
                foods_per_100.append(100.0 * float(info["score"]) / max(1.0, float(t)))
                traces.append(
                    {
                        "episode_index": ep,
                        "score": int(info["score"]),
                        "opponent_score": int(info.get("opponent_score", 0)),
                        "reward": float(ep_reward),
                        "steps": int(t),
                        "trajectory": traj,
                    }
                )
                break
        env.close()
    if traces:
        best_idx = max(range(len(traces)), key=lambda i: (traces[i]["score"], traces[i]["reward"]))
    else:
        best_idx = 0

    metrics = {
        "eval_avg_score": float(np.mean(scores)) if scores else 0.0,
        "eval_avg_reward": float(np.mean(rewards_list)) if rewards_list else 0.0,
        "eval_avg_steps": float(np.mean(steps_list)) if steps_list else 0.0,
        "eval_avg_margin": float(np.mean(margins)) if margins else 0.0,
        "eval_food_per_100_steps": float(np.mean(foods_per_100)) if foods_per_100 else 0.0,
        "eval_step_limit_rate": float(np.mean([1.0 if r == "step_limit" else 0.0 for r in terminal_reasons])) if terminal_reasons else 0.0,
        "eval_starvation_rate": float(np.mean([1.0 if r == "starvation" else 0.0 for r in terminal_reasons])) if terminal_reasons else 0.0,
        "eval_best_score": int(np.max(scores)) if scores else 0,
    }
    picked = {
        "first": traces[0] if traces else None,
        "best": traces[best_idx] if traces else None,
        "last": traces[-1] if traces else None,
    }
    return metrics, picked


def save_eval_replays(replay_dir: str, episode: int, picked: dict) -> None:
    os.makedirs(replay_dir, exist_ok=True)
    for key in ("first", "best", "last"):
        data = picked.get(key)
        if data is None:
            continue
        path = os.path.join(replay_dir, f"eval_ep{episode:05d}_{key}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)


def audit_train_eval_alignment(cfg: TrainPPOConfig, curriculum_params: dict | None) -> None:
    train_env = SnakeEnv(
        grid_size=cfg.grid_size,
        seed=cfg.seed + 111,
        max_steps_factor=cfg.max_steps_factor,
        loop_visit_threshold=cfg.loop_visit_threshold,
        loop_visit_penalty=cfg.loop_visit_penalty,
        food_reward=cfg.food_reward,
        death_penalty=cfg.death_penalty,
        distance_reward_toward=cfg.distance_reward_toward,
        distance_penalty_away=cfg.distance_penalty_away,
        stagnation_penalty=cfg.stagnation_penalty,
        survival_reward=cfg.survival_reward,
        idle_step_coeff=cfg.idle_step_coeff,
        hunger_exp_base=cfg.hunger_exp_base,
        hunger_exp_gamma=cfg.hunger_exp_gamma,
        hunger_exp_max=cfg.hunger_exp_max,
        step_limit_penalty=cfg.step_limit_penalty,
        starvation_steps_factor=cfg.starvation_steps_factor,
        starvation_penalty=cfg.starvation_penalty,
        wall_follow_threshold=cfg.wall_follow_threshold,
        wall_follow_penalty=cfg.wall_follow_penalty,
        obstacle_count=cfg.obstacle_count,
        moving_obstacles=cfg.moving_obstacles,
        obstacle_move_period=cfg.obstacle_move_period,
        moving_food=cfg.moving_food,
        food_move_prob=cfg.food_move_prob,
        food_spawn_radius=None,
        random_start=cfg.train_random_start,
        opponent_mode=("heuristic" if cfg.self_play and cfg.self_play_mode == "heuristic" else ("frozen" if cfg.self_play else "none")),
        opponent_food_penalty=cfg.opponent_food_penalty,
        opponent_random_prob=cfg.opponent_heuristic_random_prob,
        terminal_win_reward=cfg.terminal_win_reward,
        terminal_loss_penalty=cfg.terminal_loss_penalty,
    )
    eval_env = SnakeEnv(
        grid_size=cfg.grid_size,
        seed=cfg.seed + 222,
        max_steps_factor=cfg.max_steps_factor,
        loop_visit_threshold=cfg.loop_visit_threshold,
        loop_visit_penalty=cfg.loop_visit_penalty,
        food_reward=cfg.food_reward,
        death_penalty=cfg.death_penalty,
        distance_reward_toward=cfg.distance_reward_toward,
        distance_penalty_away=cfg.distance_penalty_away,
        stagnation_penalty=cfg.stagnation_penalty,
        survival_reward=cfg.survival_reward,
        idle_step_coeff=cfg.idle_step_coeff,
        hunger_exp_base=cfg.hunger_exp_base,
        hunger_exp_gamma=cfg.hunger_exp_gamma,
        hunger_exp_max=cfg.hunger_exp_max,
        step_limit_penalty=cfg.step_limit_penalty,
        starvation_steps_factor=cfg.starvation_steps_factor,
        starvation_penalty=cfg.starvation_penalty,
        wall_follow_threshold=cfg.wall_follow_threshold,
        wall_follow_penalty=cfg.wall_follow_penalty,
        obstacle_count=cfg.obstacle_count,
        moving_obstacles=cfg.moving_obstacles,
        obstacle_move_period=cfg.obstacle_move_period,
        moving_food=cfg.moving_food,
        food_move_prob=cfg.food_move_prob,
        food_spawn_radius=None,
        random_start=cfg.eval_random_start,
        opponent_mode=("heuristic" if cfg.self_play and cfg.self_play_mode == "heuristic" else ("frozen" if cfg.self_play else "none")),
        opponent_food_penalty=cfg.opponent_food_penalty,
        opponent_random_prob=cfg.opponent_heuristic_random_prob,
        terminal_win_reward=cfg.terminal_win_reward,
        terminal_loss_penalty=cfg.terminal_loss_penalty,
    )

    if curriculum_params:
        train_env.set_curriculum(**curriculum_params)
        eval_env.set_curriculum(**curriculum_params)

    keys = [
        "max_steps_factor",
        "loop_visit_threshold",
        "loop_visit_penalty",
        "food_reward",
        "death_penalty",
        "distance_reward_toward",
        "distance_penalty_away",
        "stagnation_penalty",
        "survival_reward",
        "hunger_exp_base",
        "hunger_exp_gamma",
        "hunger_exp_max",
        "step_limit_penalty",
        "starvation_steps_factor",
        "starvation_penalty",
        "wall_follow_threshold",
        "wall_follow_penalty",
        "obstacle_count",
        "moving_obstacles",
        "obstacle_move_period",
        "moving_food",
        "food_move_prob",
        "opponent_mode",
        "opponent_food_penalty",
        "opponent_random_prob",
        "terminal_win_reward",
        "terminal_loss_penalty",
        "obs_channels",
    ]
    mismatches = [k for k in keys if getattr(train_env, k) != getattr(eval_env, k)]
    train_obs = train_env.reset()
    eval_obs = eval_env.reset()
    if train_obs.shape != eval_obs.shape:
        mismatches.append("observation_shape")

    if mismatches:
        print(f"[audit] train/eval mismatch detected: {sorted(set(mismatches))}")
    else:
        print("[audit] train/eval reward+state config aligned.")
    if cfg.obs_noise_std > 0.0:
        print(f"[audit] training-only observation noise enabled (std={cfg.obs_noise_std:.4f}); eval uses clean observations.")

    train_env.close()
    eval_env.close()


def train(cfg: TrainPPOConfig) -> str:
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    project_root = os.path.abspath(cfg.out_dir)
    ckpt_dir = os.path.join(project_root, "checkpoints")
    log_dir = os.path.join(project_root, "logs")
    replay_dir = os.path.join(project_root, "replays")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(replay_dir, exist_ok=True)

    ppo_cfg = PPOConfig(
        lr=cfg.lr,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_coef=cfg.clip_coef,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        rollout_steps=cfg.rollout_steps,
        n_envs=cfg.n_envs,
        update_epochs=cfg.update_epochs,
        minibatch_size=cfg.minibatch_size,
        obs_channels=4,
    )

    agent = PPOAgent(grid_size=cfg.grid_size, device=device, cfg=ppo_cfg)
    vec = SyncVecSnake(cfg)
    ppo_cfg.obs_channels = vec.obs_channels

    if ppo_cfg.obs_channels != 4:
        # Rebuild with actual env channel count.
        ppo_cfg.obs_channels = vec.obs_channels
        agent = PPOAgent(grid_size=cfg.grid_size, device=device, cfg=ppo_cfg)

    curriculum = CurriculumManager(promote_streak=cfg.curriculum_promote_streak)
    entropy_ctrl = AdaptiveEntropyController(
        base_start=cfg.ent_coef,
        base_end=cfg.ent_coef_end,
        min_coef=cfg.entropy_min,
        max_coef=cfg.entropy_max,
        up_step=cfg.entropy_up_step,
        down_step=cfg.entropy_down_step,
        plateau_patience=cfg.entropy_plateau_patience,
    )

    opponent_agent: PPOAgent | None = None
    if cfg.self_play and cfg.self_play_mode == "last_best":
        opponent_cfg = PPOConfig(**{**ppo_cfg.__dict__})
        opponent_agent = PPOAgent(grid_size=cfg.grid_size, device=device, cfg=opponent_cfg)

    def should_use_last_best_opponent() -> bool:
        if not (cfg.self_play and cfg.self_play_mode == "last_best"):
            return False
        if opponent_agent is None:
            return False
        if episodes_done < int(cfg.self_play_warmup_episodes):
            return False
        if best_eval_score < float(cfg.self_play_min_eval_score):
            return False
        return True

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(log_dir, f"train_ppo_{run_id}.csv")

    obs, aux = vec.reset()
    episodes_done = 0
    updates = 0
    best_eval_score = -10**9
    best_model_score = -10**9
    severe_regression_streak = 0
    best_eval_by_level: dict[int, float] = {}
    next_eval_episode = max(1, int(cfg.eval_every))
    next_save_episode = max(1, int(cfg.save_every))

    recent_reward = deque(maxlen=200)
    recent_score = deque(maxlen=200)
    recent_len = deque(maxlen=200)

    ep_reward_acc = np.zeros((cfg.n_envs,), dtype=np.float32)
    env_steps = 0

    resumed_from = ""
    if cfg.resume:
        resumed_from = os.path.abspath(cfg.resume)
        extra = agent.load(cfg.resume, strict=False)
        if cfg.resume_reset_optim:
            agent.optim = torch.optim.Adam(agent.net.parameters(), lr=cfg.lr)
        state = extra.get("train_state", {}) if isinstance(extra, dict) else {}
        episodes_done = int(state.get("episodes_done", episodes_done))
        updates = int(state.get("updates", updates))
        env_steps = int(state.get("env_steps", env_steps))
        best_eval_score = float(state.get("best_eval_score", best_eval_score))
        best_model_score = float(state.get("best_model_score", best_eval_score))
        next_eval_episode = int(state.get("next_eval_episode", next_eval_episode))
        next_save_episode = int(state.get("next_save_episode", next_save_episode))
        curriculum.level = int(state.get("curriculum_level", curriculum.level))
        curriculum.streak = int(state.get("curriculum_streak", curriculum.streak))
        curriculum.fail_streak = int(state.get("curriculum_fail_streak", curriculum.fail_streak))
        curriculum.eval_ema = float(state.get("curriculum_eval_ema", curriculum.eval_ema))
        curriculum.level_entry_episode = int(state.get("curriculum_level_entry_episode", curriculum.level_entry_episode))
        entropy_ctrl.current = float(state.get("entropy_current", entropy_ctrl.current))

    if opponent_agent is not None:
        best_path = os.path.join(ckpt_dir, "best.pt")
        if os.path.exists(best_path):
            opponent_agent.load(best_path, strict=False)

    def maybe_add_obs_noise(obs_in: np.ndarray) -> np.ndarray:
        std = float(max(0.0, cfg.obs_noise_std))
        if std <= 0.0:
            return obs_in
        noise = np.random.normal(loc=0.0, scale=std, size=obs_in.shape).astype(np.float32)
        return np.clip(obs_in + noise, 0.0, 1.0).astype(np.float32)

    audit_done = False

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode",
                "updates",
                "env_steps",
                "train_reward_avg200",
                "train_score_avg200",
                "train_length_avg200",
                "loss_pi",
                "loss_v",
                "entropy",
                "clipfrac",
                "ent_coef_now",
                "curriculum_level",
                "curriculum_streak",
                "self_play",
                "resumed_from",
                "eval_avg_score",
                "eval_avg_reward",
                "eval_avg_steps",
                "eval_avg_margin",
                "eval_food_per_100_steps",
                "eval_step_limit_rate",
                "eval_starvation_rate",
            ]
        )

        while episodes_done < cfg.episodes:
            progress = episodes_done / max(1, cfg.episodes)
            if cfg.randomize_train_seeds and updates % max(1, int(cfg.reseed_every_updates)) == 0:
                seed_base = int(np.random.randint(0, max(2, int(cfg.reseed_span))))
                vec.reseed_all(seed_base, span=cfg.reseed_span)
            if cfg.use_curriculum:
                if curriculum.level > 0 and (cfg.curriculum_prev_mix_prob > 0.0 or cfg.curriculum_history_mix_prob > 0.0):
                    p_prev = float(np.clip(cfg.curriculum_prev_mix_prob, 0.0, 1.0))
                    p_hist = float(np.clip(cfg.curriculum_history_mix_prob, 0.0, 1.0))
                    if p_prev + p_hist > 1.0:
                        s = p_prev + p_hist
                        p_prev /= s
                        p_hist /= s

                    current = curriculum.current_params()
                    previous = curriculum.params_for_level(curriculum.level - 1)
                    params_per_env: list[dict] = []
                    for _ in range(cfg.n_envs):
                        r = random.random()
                        if r < p_hist and curriculum.level >= 1:
                            hist_level = random.randint(0, curriculum.level - 1)
                            params_per_env.append(curriculum.params_for_level(hist_level))
                        elif r < (p_hist + p_prev):
                            params_per_env.append(previous)
                        else:
                            params_per_env.append(current)
                    vec.apply_curriculum_params_per_env(params_per_env)
                else:
                    vec.apply_curriculum_params(curriculum.current_params())

            t_max = cfg.rollout_steps
            n_envs = cfg.n_envs
            obs_buf = np.zeros((t_max, n_envs, cfg.grid_size, cfg.grid_size, vec.obs_channels), dtype=np.float32)
            aux_buf = np.zeros((t_max, n_envs, ppo_cfg.aux_dim), dtype=np.float32)
            actions_buf = np.zeros((t_max, n_envs), dtype=np.int64)
            logp_buf = np.zeros((t_max, n_envs), dtype=np.float32)
            rewards_buf = np.zeros((t_max, n_envs), dtype=np.float32)
            dones_buf = np.zeros((t_max, n_envs), dtype=np.float32)
            values_buf = np.zeros((t_max, n_envs), dtype=np.float32)

            collected = 0
            for t in range(t_max):
                obs_in = maybe_add_obs_noise(obs)
                a, lp, v = agent.act(obs_in, aux, deterministic=False)
                use_best_mask = None
                if cfg.self_play and cfg.self_play_mode == "last_best":
                    if should_use_last_best_opponent():
                        use_best_mask = (np.random.random(size=(n_envs,)) < float(cfg.self_play_best_prob)).astype(bool)
                    else:
                        use_best_mask = np.zeros((n_envs,), dtype=bool)
                next_obs, next_aux, r, d, infos = vec.step(
                    a,
                    opponent_agent=opponent_agent,
                    use_best_mask=use_best_mask,
                )

                ep_reward_acc += r

                obs_buf[t] = obs_in
                aux_buf[t] = aux
                actions_buf[t] = a
                logp_buf[t] = lp
                rewards_buf[t] = r
                dones_buf[t] = d
                values_buf[t] = v

                env_steps += n_envs
                collected += 1
                obs = next_obs
                aux = next_aux

                for i in range(n_envs):
                    if d[i] > 0.5:
                        episodes_done += 1
                        recent_reward.append(float(ep_reward_acc[i]))
                        recent_score.append(float(infos[i]["score"]))
                        recent_len.append(float(infos[i]["length"]))
                        ep_reward_acc[i] = 0.0
                        if episodes_done >= cfg.episodes:
                            break
                if episodes_done >= cfg.episodes:
                    break

            next_value = agent.value(maybe_add_obs_noise(obs), aux)
            obs_b = obs_buf[:collected]
            aux_b = aux_buf[:collected]
            actions_b = actions_buf[:collected]
            logp_b = logp_buf[:collected]
            rewards_b = rewards_buf[:collected]
            dones_b = dones_buf[:collected]
            values_b = values_buf[:collected]

            adv, ret = compute_gae(rewards_b, dones_b, values_b, next_value, cfg.gamma, cfg.gae_lambda)

            flat = {
                "obs": obs_b.reshape(-1, cfg.grid_size, cfg.grid_size, vec.obs_channels),
                "aux": aux_b.reshape(-1, ppo_cfg.aux_dim),
                "actions": actions_b.reshape(-1),
                "logprobs": logp_b.reshape(-1),
                "advantages": adv.reshape(-1),
                "returns": ret.reshape(-1),
                "values": values_b.reshape(-1),
            }

            if cfg.use_adaptive_entropy:
                ent_coef_now = entropy_ctrl.get(progress)
            else:
                ent_coef_now = cfg.ent_coef + progress * (cfg.ent_coef_end - cfg.ent_coef)
            if cfg.use_curriculum and curriculum.level >= 3:
                ent_coef_now = max(float(ent_coef_now), float(cfg.pro_entropy_floor))
            losses = agent.update(flat, ent_coef_override=ent_coef_now)
            updates += 1

            eval_metrics = {
                "eval_avg_score": "",
                "eval_avg_reward": "",
                "eval_avg_steps": "",
                "eval_avg_margin": "",
                "eval_food_per_100_steps": "",
                "eval_step_limit_rate": "",
                "eval_starvation_rate": "",
            }
            while episodes_done >= next_eval_episode:
                if not audit_done:
                    audit_train_eval_alignment(cfg, curriculum.current_params() if cfg.use_curriculum else None)
                    audit_done = True
                metrics_batches: list[dict] = []
                picked = {"first": None, "best": None, "last": None}
                batches = max(1, int(cfg.eval_seed_batches))
                eps_per_batch = max(1, int(np.ceil(float(cfg.eval_episodes) / float(batches))))
                for b in range(batches):
                    mb, picked_b = evaluate(
                        agent,
                        cfg,
                        eps_per_batch,
                        curriculum_params=(curriculum.current_params() if cfg.use_curriculum else None),
                        opponent_agent=(opponent_agent if should_use_last_best_opponent() else None),
                        seed_offset=next_eval_episode + b * 10007,
                    )
                    metrics_batches.append(mb)
                    if b == 0:
                        picked = picked_b

                m = {
                    "eval_avg_score": float(np.mean([x["eval_avg_score"] for x in metrics_batches])),
                    "eval_avg_reward": float(np.mean([x["eval_avg_reward"] for x in metrics_batches])),
                    "eval_avg_steps": float(np.mean([x["eval_avg_steps"] for x in metrics_batches])),
                    "eval_avg_margin": float(np.mean([x["eval_avg_margin"] for x in metrics_batches])),
                    "eval_food_per_100_steps": float(np.mean([x["eval_food_per_100_steps"] for x in metrics_batches])),
                    "eval_step_limit_rate": float(np.mean([x["eval_step_limit_rate"] for x in metrics_batches])),
                    "eval_starvation_rate": float(np.mean([x["eval_starvation_rate"] for x in metrics_batches])),
                    "eval_best_score": int(np.max([x["eval_best_score"] for x in metrics_batches])),
                }
                eval_metrics = m
                save_eval_replays(replay_dir, next_eval_episode, picked)

                promoted = False
                if cfg.use_curriculum:
                    promoted, _ = curriculum.on_eval(
                        float(m["eval_avg_score"]),
                        episodes_done,
                        eval_avg_margin=float(m.get("eval_avg_margin", 0.0)),
                        eval_step_limit_rate=float(m.get("eval_step_limit_rate", 1.0)),
                        eval_food_per_100_steps=float(m.get("eval_food_per_100_steps", 0.0)),
                    )

                if cfg.use_adaptive_entropy:
                    entropy_ctrl.update_from_eval(float(m["eval_avg_score"]), float(losses.get("entropy", 0.0)))

                model_score = (
                    float(m["eval_avg_score"])
                    + 0.1 * float(m.get("eval_avg_margin", 0.0))
                    + 0.02 * float(m.get("eval_food_per_100_steps", 0.0))
                    - 0.25 * float(m.get("eval_step_limit_rate", 0.0))
                    - 0.15 * float(m.get("eval_starvation_rate", 0.0))
                )
                if model_score > best_model_score:
                    best_model_score = model_score
                    best_eval_score = m["eval_avg_score"]
                    severe_regression_streak = 0
                    train_state = {
                        "episodes_done": episodes_done,
                        "updates": updates,
                        "env_steps": env_steps,
                        "best_eval_score": best_eval_score,
                        "next_eval_episode": next_eval_episode,
                        "next_save_episode": next_save_episode,
                        "curriculum_level": curriculum.level,
                        "curriculum_streak": curriculum.streak,
                        "curriculum_fail_streak": curriculum.fail_streak,
                        "curriculum_eval_ema": curriculum.eval_ema,
                        "curriculum_level_entry_episode": curriculum.level_entry_episode,
                        "entropy_current": entropy_ctrl.current,
                    }
                    agent.save(
                        os.path.join(ckpt_dir, "best.pt"),
                        extra={"episode": next_eval_episode, "eval": m, "cfg": asdict(cfg), "train_state": train_state},
                    )
                    if opponent_agent is not None:
                        opponent_agent.load(os.path.join(ckpt_dir, "best.pt"), strict=False)

                cur_level = int(curriculum.level)
                cur_eval = float(m["eval_avg_score"])
                prev_level_best = float(best_eval_by_level.get(cur_level, -1e9))
                if cur_eval > prev_level_best:
                    best_eval_by_level[cur_level] = cur_eval

                level_best = float(best_eval_by_level.get(cur_level, cur_eval))
                if cfg.use_regression_rollback:
                    if level_best > 0.0 and cur_eval < 0.60 * level_best:
                        severe_regression_streak += 1
                    else:
                        severe_regression_streak = max(0, severe_regression_streak - 1)

                    if severe_regression_streak >= 3:
                        best_path = os.path.join(ckpt_dir, "best.pt")
                        if os.path.exists(best_path):
                            agent.load(best_path, strict=False)
                            if opponent_agent is not None:
                                opponent_agent.load(best_path, strict=False)
                        severe_regression_streak = 0

                if promoted:
                    vec.apply_curriculum_params(curriculum.current_params())

                next_eval_episode += max(1, int(cfg.eval_every))

            while episodes_done >= next_save_episode:
                agent.save(
                    os.path.join(ckpt_dir, f"checkpoint_{next_save_episode}.pt"),
                    extra={
                        "episode": next_save_episode,
                        "best_eval_score": best_eval_score,
                        "cfg": asdict(cfg),
                        "train_state": {
                            "episodes_done": episodes_done,
                            "updates": updates,
                            "env_steps": env_steps,
                            "best_eval_score": best_eval_score,
                            "next_eval_episode": next_eval_episode,
                            "next_save_episode": next_save_episode,
                            "curriculum_level": curriculum.level,
                            "curriculum_streak": curriculum.streak,
                            "curriculum_fail_streak": curriculum.fail_streak,
                            "curriculum_eval_ema": curriculum.eval_ema,
                            "curriculum_level_entry_episode": curriculum.level_entry_episode,
                            "entropy_current": entropy_ctrl.current,
                            "best_model_score": best_model_score,
                        },
                    },
                )
                next_save_episode += max(1, int(cfg.save_every))

            writer.writerow(
                [
                    episodes_done,
                    updates,
                    env_steps,
                    float(np.mean(recent_reward)) if recent_reward else 0.0,
                    float(np.mean(recent_score)) if recent_score else 0.0,
                    float(np.mean(recent_len)) if recent_len else 0.0,
                    losses["loss_pi"],
                    losses["loss_v"],
                    losses["entropy"],
                    losses["clipfrac"],
                    ent_coef_now,
                    curriculum.level,
                    curriculum.streak,
                    int(cfg.self_play),
                    resumed_from,
                    eval_metrics["eval_avg_score"],
                    eval_metrics["eval_avg_reward"],
                    eval_metrics["eval_avg_steps"],
                    eval_metrics["eval_avg_margin"],
                    eval_metrics["eval_food_per_100_steps"],
                    eval_metrics["eval_step_limit_rate"],
                    eval_metrics["eval_starvation_rate"],
                ]
            )

            if episodes_done % cfg.log_every == 0 or episodes_done < cfg.n_envs:
                print(
                    f"[ep {episodes_done:4d}] updates={updates:4d} steps={env_steps:8d} "
                    f"score(avg200)={float(np.mean(recent_score)) if recent_score else 0.0:.3f} "
                    f"reward(avg200)={float(np.mean(recent_reward)) if recent_reward else 0.0:.3f} "
                    f"pi={losses['loss_pi']:.4f} v={losses['loss_v']:.4f} ent={losses['entropy']:.4f} ent_coef={ent_coef_now:.4f}"
                )

    final_path = os.path.join(ckpt_dir, "final.pt")
    agent.save(
        final_path,
        extra={
            "episode": episodes_done,
            "best_eval_score": best_eval_score,
            "cfg": asdict(cfg),
            "train_state": {
                "episodes_done": episodes_done,
                "updates": updates,
                "env_steps": env_steps,
                "best_eval_score": best_eval_score,
                "next_eval_episode": next_eval_episode,
                "next_save_episode": next_save_episode,
                "curriculum_level": curriculum.level,
                "curriculum_streak": curriculum.streak,
                "curriculum_fail_streak": curriculum.fail_streak,
                "curriculum_eval_ema": curriculum.eval_ema,
                "curriculum_level_entry_episode": curriculum.level_entry_episode,
                "entropy_current": entropy_ctrl.current,
                "best_model_score": best_model_score,
            },
        },
    )
    vec.close()

    print(f"Training complete. Final checkpoint: {final_path}")
    print(f"Metrics CSV: {csv_path}")
    print(f"Eval replays directory: {replay_dir}")
    return final_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train Snake with PPO (CNN actor-critic)",
        formatter_class=_HelpFormatter,
        epilog=(
            "Examples:\n"
            "  python train.py --episodes 3000 --device cuda\n"
            "  python train.py --episodes 200 --n-envs 4 --rollout-steps 128"
        ),
    )

    p.add_argument("--episodes", type=int, default=3000, help="Number of completed episodes target")
    p.add_argument("--grid-size", type=int, default=20, help="Board size N for an N x N grid")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--out-dir", type=str, default=".", help="Output directory root")
    p.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, ...")

    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.006)
    p.add_argument("--ent-coef-end", type=float, default=0.0005)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--n-envs", type=int, default=12)
    p.add_argument("--rollout-steps", type=int, default=256)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=512)

    p.add_argument("--max-steps-factor", type=int, default=100)
    p.add_argument("--loop-visit-threshold", type=int, default=3)
    p.add_argument("--loop-visit-penalty", type=float, default=-0.03)
    p.add_argument("--food-reward", type=float, default=15.0)
    p.add_argument("--death-penalty", type=float, default=-8.0)
    p.add_argument("--distance-reward-toward", type=float, default=0.2)
    p.add_argument("--distance-penalty-away", type=float, default=-0.1)
    p.add_argument("--stagnation-penalty", type=float, default=0.0)
    p.add_argument("--survival-reward", type=float, default=-0.003)
    p.add_argument("--idle-step-coeff", type=float, default=0.0)
    p.add_argument("--hunger-exp-base", type=float, default=0.003)
    p.add_argument("--hunger-exp-gamma", type=float, default=1.02)
    p.add_argument("--hunger-exp-max", type=float, default=1.2)
    p.add_argument("--step-limit-penalty", type=float, default=-8.0)
    p.add_argument("--starvation-steps-factor", type=int, default=55)
    p.add_argument("--starvation-penalty", type=float, default=-5.0)
    p.add_argument("--wall-follow-threshold", type=int, default=5)
    p.add_argument("--wall-follow-penalty", type=float, default=-0.2)
    p.add_argument("--obstacle-count", type=int, default=0)
    p.add_argument("--moving-obstacles", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--obstacle-move-period", type=int, default=8)
    p.add_argument("--moving-food", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--food-move-prob", type=float, default=0.15)
    p.add_argument("--train-random-start", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval-random-start", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--obs-noise-std", type=float, default=0.005)

    p.add_argument(
        "--use-curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable dynamic curriculum over training progress",
    )
    p.add_argument("--curriculum-promote-streak", type=int, default=3)
    p.add_argument("--curriculum-prev-mix-prob", type=float, default=0.30)
    p.add_argument("--curriculum-history-mix-prob", type=float, default=0.20)
    p.add_argument("--randomize-train-seeds", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reseed-every-updates", type=int, default=12)
    p.add_argument("--reseed-span", type=int, default=1000000)

    p.add_argument(
        "--use-adaptive-entropy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Adjust entropy coefficient based on eval stability/progress",
    )
    p.add_argument("--entropy-min", type=float, default=1.5e-3)
    p.add_argument("--entropy-max", type=float, default=2e-2)
    p.add_argument("--entropy-up-step", type=float, default=1.2e-3)
    p.add_argument("--entropy-down-step", type=float, default=4e-4)
    p.add_argument("--entropy-plateau-patience", type=int, default=2)
    p.add_argument("--pro-entropy-floor", type=float, default=0.0018)
    p.add_argument("--use-regression-rollback", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--self-play", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--self-play-mode", type=str, default="heuristic", choices=["heuristic", "last_best"])
    p.add_argument("--opponent-food-penalty", type=float, default=-2.0)
    p.add_argument("--opponent-heuristic-random-prob", type=float, default=0.15)
    p.add_argument("--terminal-win-reward", type=float, default=8.0)
    p.add_argument("--terminal-loss-penalty", type=float, default=-8.0)
    p.add_argument("--self-play-warmup-episodes", type=int, default=200)
    p.add_argument("--self-play-best-prob", type=float, default=0.7)
    p.add_argument("--self-play-min-eval-score", type=float, default=1.0)

    p.add_argument("--resume", type=str, default="")
    p.add_argument("--resume-reset-optim", action="store_true")

    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--eval-seed-batches", type=int, default=1)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=10)
    return p


def main() -> None:
    a = build_parser().parse_args()
    cfg = TrainPPOConfig(
        episodes=a.episodes,
        grid_size=a.grid_size,
        seed=a.seed,
        out_dir=a.out_dir,
        device=a.device,
        lr=a.lr,
        gamma=a.gamma,
        gae_lambda=a.gae_lambda,
        clip_coef=a.clip_coef,
        ent_coef=a.ent_coef,
        ent_coef_end=a.ent_coef_end,
        vf_coef=a.vf_coef,
        max_grad_norm=a.max_grad_norm,
        n_envs=a.n_envs,
        rollout_steps=a.rollout_steps,
        update_epochs=a.update_epochs,
        minibatch_size=a.minibatch_size,
        max_steps_factor=a.max_steps_factor,
        loop_visit_threshold=a.loop_visit_threshold,
        loop_visit_penalty=a.loop_visit_penalty,
        food_reward=a.food_reward,
        death_penalty=a.death_penalty,
        distance_reward_toward=a.distance_reward_toward,
        distance_penalty_away=a.distance_penalty_away,
        stagnation_penalty=a.stagnation_penalty,
        survival_reward=a.survival_reward,
        idle_step_coeff=a.idle_step_coeff,
        hunger_exp_base=a.hunger_exp_base,
        hunger_exp_gamma=a.hunger_exp_gamma,
        hunger_exp_max=a.hunger_exp_max,
        step_limit_penalty=a.step_limit_penalty,
        starvation_steps_factor=a.starvation_steps_factor,
        starvation_penalty=a.starvation_penalty,
        wall_follow_threshold=a.wall_follow_threshold,
        wall_follow_penalty=a.wall_follow_penalty,
        obstacle_count=a.obstacle_count,
        moving_obstacles=a.moving_obstacles,
        obstacle_move_period=a.obstacle_move_period,
        moving_food=a.moving_food,
        food_move_prob=a.food_move_prob,
        train_random_start=a.train_random_start,
        eval_random_start=a.eval_random_start,
        obs_noise_std=a.obs_noise_std,
        use_curriculum=a.use_curriculum,
        curriculum_promote_streak=a.curriculum_promote_streak,
        curriculum_prev_mix_prob=a.curriculum_prev_mix_prob,
        curriculum_history_mix_prob=a.curriculum_history_mix_prob,
        randomize_train_seeds=a.randomize_train_seeds,
        reseed_every_updates=a.reseed_every_updates,
        reseed_span=a.reseed_span,
        use_adaptive_entropy=a.use_adaptive_entropy,
        entropy_min=a.entropy_min,
        entropy_max=a.entropy_max,
        entropy_up_step=a.entropy_up_step,
        entropy_down_step=a.entropy_down_step,
        entropy_plateau_patience=a.entropy_plateau_patience,
        pro_entropy_floor=a.pro_entropy_floor,
        use_regression_rollback=a.use_regression_rollback,
        self_play=a.self_play,
        self_play_mode=a.self_play_mode,
        opponent_food_penalty=a.opponent_food_penalty,
        opponent_heuristic_random_prob=a.opponent_heuristic_random_prob,
        terminal_win_reward=a.terminal_win_reward,
        terminal_loss_penalty=a.terminal_loss_penalty,
        self_play_warmup_episodes=a.self_play_warmup_episodes,
        self_play_best_prob=a.self_play_best_prob,
        self_play_min_eval_score=a.self_play_min_eval_score,
        resume=a.resume,
        resume_reset_optim=a.resume_reset_optim,
        eval_every=a.eval_every,
        eval_episodes=a.eval_episodes,
        eval_seed_batches=a.eval_seed_batches,
        save_every=a.save_every,
        log_every=a.log_every,
    )
    train(cfg)


if __name__ == "__main__":
    main()
