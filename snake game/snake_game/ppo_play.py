from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from .env_v2 import SnakeEnv
from .ppo_agent import PPOAgent, PPOConfig


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def run(
    checkpoint: str,
    episodes: int = 3,
    grid_size: int = 20,
    fps: int = 12,
    max_steps: int = 4000,
    device: str = "auto",
    deterministic: bool = True,
    opponent_mode: str = "none",
    opponent_food_penalty: float = -0.10,
    opponent_random_prob: float = 0.80,
    obstacle_count: int = 0,
    moving_obstacles: bool = False,
    obstacle_move_period: int = 8,
    moving_food: bool = False,
    food_move_prob: float = 0.15,
    terminal_win_reward: float = 3.0,
    terminal_loss_penalty: float = -3.0,
    starvation_steps_factor: int = 60,
    starvation_penalty: float = -6.0,
    wall_follow_threshold: int = 6,
    wall_follow_penalty: float = -0.12,
    curriculum: bool = False,
) -> None:
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)

    env = SnakeEnv(
        grid_size=grid_size,
        opponent_mode=opponent_mode,
        opponent_food_penalty=opponent_food_penalty,
        opponent_random_prob=opponent_random_prob,
        obstacle_count=obstacle_count,
        moving_obstacles=moving_obstacles,
        obstacle_move_period=obstacle_move_period,
        moving_food=moving_food,
        food_move_prob=food_move_prob,
        terminal_win_reward=terminal_win_reward,
        terminal_loss_penalty=terminal_loss_penalty,
        starvation_steps_factor=starvation_steps_factor,
        starvation_penalty=starvation_penalty,
        wall_follow_threshold=wall_follow_threshold,
        wall_follow_penalty=wall_follow_penalty,
    )
    play_cfg = PPOConfig(obs_channels=env.obs_channels)
    agent = PPOAgent(grid_size=grid_size, device=dev, cfg=play_cfg)

    if checkpoint and os.path.exists(checkpoint):
        agent.load(checkpoint, strict=False)
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print("Checkpoint not found. Running with random-initialized model.")

    level_specs = [
        (0, {"opponent_mode": "none", "obstacle_count": 0, "moving_obstacles": False, "moving_food": False}),
        (1, {"opponent_mode": "none", "obstacle_count": 6, "moving_obstacles": False, "moving_food": False}),
        (2, {"opponent_mode": "none", "obstacle_count": 8, "moving_obstacles": True, "obstacle_move_period": 10, "moving_food": False}),
        (3, {"opponent_mode": "none", "obstacle_count": 10, "moving_obstacles": True, "obstacle_move_period": 9, "moving_food": True, "food_move_prob": 0.08}),
        (4, {"opponent_mode": "heuristic", "obstacle_count": 10, "moving_obstacles": True, "obstacle_move_period": 7, "moving_food": True, "food_move_prob": 0.10, "opponent_food_penalty": -0.10, "opponent_random_prob": 0.80}),
    ]

    def run_one_episode(global_ep: int, level_tag: str) -> None:
        s = env.reset()
        total_r = 0.0
        for t in range(max_steps):
            env.render(fps=fps)
            aux = env.aux_features()
            a, _, _ = agent.act(np.expand_dims(s, axis=0), np.expand_dims(aux, axis=0), deterministic=deterministic)
            s, r, done, truncated, info = env.step(int(a[0]))
            total_r += r
            if done or truncated:
                print(
                    f"[play {level_tag} ep {global_ep}] steps={t+1} score={info['score']} "
                    f"length={info['length']} reward={total_r:.2f} reason={info['reason']}"
                )
                break
        time.sleep(0.3)

    if curriculum:
        print("Curriculum play mode: running each level from 0 to 4")
        for level, params in level_specs:
            env.set_curriculum(**params)
            print(f"\n=== Level {level} ===")
            for ep in range(1, episodes + 1):
                run_one_episode(ep, f"L{level}")
    else:
        for ep in range(1, episodes + 1):
            run_one_episode(ep, "single")

    env.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Play Snake using a trained PPO checkpoint",
        formatter_class=_HelpFormatter,
        epilog=(
            "Examples:\n"
            "  python play.py --checkpoint ./checkpoints/final.pt\n"
            "  python play.py --checkpoint ./checkpoints/best.pt --episodes 5 --fps 20"
        ),
    )
    p.add_argument("--checkpoint", type=str, default="./checkpoints/final.pt")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--grid-size", type=int, default=20)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--stochastic", action="store_true", help="Sample policy action instead of greedy action")
    p.add_argument("--opponent-mode", type=str, default="none", choices=["none", "heuristic"])
    p.add_argument("--opponent-food-penalty", type=float, default=-0.10)
    p.add_argument("--opponent-random-prob", type=float, default=0.80)
    p.add_argument("--obstacle-count", type=int, default=0)
    p.add_argument("--moving-obstacles", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--obstacle-move-period", type=int, default=8)
    p.add_argument("--moving-food", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--food-move-prob", type=float, default=0.15)
    p.add_argument("--terminal-win-reward", type=float, default=3.0)
    p.add_argument("--terminal-loss-penalty", type=float, default=-3.0)
    p.add_argument("--starvation-steps-factor", type=int, default=60)
    p.add_argument("--starvation-penalty", type=float, default=-6.0)
    p.add_argument("--wall-follow-threshold", type=int, default=6)
    p.add_argument("--wall-follow-penalty", type=float, default=-0.12)
    p.add_argument("--curriculum", action="store_true", help="Play each curriculum level sequentially; runs --episodes at every level")
    return p


def main() -> None:
    a = build_parser().parse_args()
    run(
        checkpoint=a.checkpoint,
        episodes=a.episodes,
        grid_size=a.grid_size,
        fps=a.fps,
        max_steps=a.max_steps,
        device=a.device,
        deterministic=not a.stochastic,
        opponent_mode=a.opponent_mode,
        opponent_food_penalty=a.opponent_food_penalty,
        opponent_random_prob=a.opponent_random_prob,
        obstacle_count=a.obstacle_count,
        moving_obstacles=a.moving_obstacles,
        obstacle_move_period=a.obstacle_move_period,
        moving_food=a.moving_food,
        food_move_prob=a.food_move_prob,
        terminal_win_reward=a.terminal_win_reward,
        terminal_loss_penalty=a.terminal_loss_penalty,
        starvation_steps_factor=a.starvation_steps_factor,
        starvation_penalty=a.starvation_penalty,
        wall_follow_threshold=a.wall_follow_threshold,
        wall_follow_penalty=a.wall_follow_penalty,
        curriculum=a.curriculum,
    )


if __name__ == "__main__":
    main()
