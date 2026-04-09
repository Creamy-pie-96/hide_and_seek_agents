from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import torch

from .agent import AgentConfig, DQNAgent
from .env import SnakeEnv
from .replay import ReplayBuffer


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


@dataclass
class TrainConfig:
    episodes: int = 1500
    grid_size: int = 20
    max_steps_per_episode: int = 4000

    # DQN
    lr: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 64
    replay_capacity: int = 100_000
    warmup_steps: int = 2000
    target_update_steps: int = 1000

    # Epsilon schedule
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 80_000

    # Environment reward shaping
    max_steps_factor: int = 100
    loop_visit_threshold: int = 3
    loop_visit_penalty: float = -0.10
    food_reward: float = 10.0
    death_penalty: float = -10.0
    distance_reward_toward: float = 0.1
    distance_penalty_away: float = -0.05
    idle_step_coeff: float = 0.001
    hunger_exp_base: float = 0.0
    hunger_exp_gamma: float = 1.005
    hunger_exp_max: float = 1.0
    step_limit_penalty: float = 0.0

    # IO
    save_every: int = 50
    log_every: int = 10
    out_dir: str = "."
    seed: int = 42
    device: str = "auto"


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


def linear_epsilon(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return eps_end
    frac = min(1.0, step / float(decay_steps))
    return float(eps_start + frac * (eps_end - eps_start))


def train(cfg: TrainConfig) -> str:
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    project_root = os.path.abspath(cfg.out_dir)
    ckpt_dir = os.path.join(project_root, "checkpoints")
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = SnakeEnv(
        grid_size=cfg.grid_size,
        seed=cfg.seed,
        max_steps_factor=cfg.max_steps_factor,
        loop_visit_threshold=cfg.loop_visit_threshold,
        loop_visit_penalty=cfg.loop_visit_penalty,
        food_reward=cfg.food_reward,
        death_penalty=cfg.death_penalty,
        distance_reward_toward=cfg.distance_reward_toward,
        distance_penalty_away=cfg.distance_penalty_away,
        idle_step_coeff=cfg.idle_step_coeff,
        hunger_exp_base=cfg.hunger_exp_base,
        hunger_exp_gamma=cfg.hunger_exp_gamma,
        hunger_exp_max=cfg.hunger_exp_max,
        step_limit_penalty=cfg.step_limit_penalty,
    )
    agent = DQNAgent(
        grid_size=cfg.grid_size,
        device=device,
        cfg=AgentConfig(lr=cfg.lr, gamma=cfg.gamma, target_update_steps=cfg.target_update_steps),
    )
    buffer = ReplayBuffer(cfg.replay_capacity, (cfg.grid_size, cfg.grid_size, 3), device)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(log_dir, f"train_{run_id}.csv")

    global_step = 0
    best_score = -10**9

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "steps", "epsilon", "reward", "score", "length", "loss_avg"])

        for ep in range(1, cfg.episodes + 1):
            state = env.reset()
            ep_reward = 0.0
            ep_losses: list[float] = []

            for _ in range(cfg.max_steps_per_episode):
                eps = linear_epsilon(global_step, cfg.eps_start, cfg.eps_end, cfg.eps_decay_steps)
                action = agent.act(state, eps)
                next_state, reward, done, truncated, info = env.step(action)

                terminal = bool(done or truncated)
                buffer.push(state, action, reward, next_state, terminal)

                state = next_state
                ep_reward += reward
                global_step += 1

                if len(buffer) >= max(cfg.batch_size, cfg.warmup_steps):
                    loss = agent.optimize(buffer.sample(cfg.batch_size))
                    ep_losses.append(loss)

                if terminal:
                    break

            avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
            score = int(info["score"])
            length = int(info["length"])

            writer.writerow([ep, global_step, eps, ep_reward, score, length, avg_loss])

            if ep % cfg.log_every == 0 or ep == 1:
                print(
                    f"[ep {ep:4d}] steps={global_step:7d} eps={eps:.3f} "
                    f"reward={ep_reward:8.3f} score={score:3d} len={length:3d} loss={avg_loss:.5f}"
                )

            if score > best_score:
                best_score = score
                best_path = os.path.join(ckpt_dir, "best.pt")
                agent.save(best_path, extra={"episode": ep, "score": score, "cfg": asdict(cfg)})

            if ep % cfg.save_every == 0:
                ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{ep}.pt")
                agent.save(ckpt_path, extra={"episode": ep, "score": score, "cfg": asdict(cfg)})

    final_path = os.path.join(ckpt_dir, "final.pt")
    agent.save(final_path, extra={"episode": cfg.episodes, "best_score": best_score, "cfg": asdict(cfg)})
    env.close()

    print(f"Training complete. Final checkpoint: {final_path}")
    print(f"Metrics CSV: {csv_path}")
    return final_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train CNN-DQN Snake agent",
        formatter_class=_HelpFormatter,
        epilog=(
            "Examples:\n"
            "  python train.py --episodes 1500\n"
            "  python train.py --episodes 5 --max-steps-per-episode 200 --warmup-steps 64"
        ),
    )
    p.add_argument("--episodes", type=int, default=1500, help="Number of episodes to train")
    p.add_argument(
        "--grid-size",
        type=int,
        default=20,
        help="Board size N for an N x N grid",
    )
    p.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=4000,
        help="Hard cap on steps in one episode",
    )

    p.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    p.add_argument("--batch-size", type=int, default=64, help="Minibatch size for optimization")
    p.add_argument(
        "--replay-capacity",
        type=int,
        default=100_000,
        help="Replay buffer maximum number of transitions",
    )
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=2000,
        help="Minimum transitions before training starts",
    )
    p.add_argument(
        "--target-update-steps",
        type=int,
        default=1000,
        help="How often (optimizer steps) to copy online -> target network",
    )

    p.add_argument("--eps-start", type=float, default=1.0, help="Initial epsilon for epsilon-greedy")
    p.add_argument("--eps-end", type=float, default=0.01, help="Final epsilon after decay")
    p.add_argument(
        "--eps-decay-steps",
        type=int,
        default=80_000,
        help="Number of environment steps over which epsilon decays linearly",
    )

    p.add_argument("--max-steps-factor", type=int, default=100, help="Episode truncation factor: max_steps = factor * snake_length")
    p.add_argument("--loop-visit-threshold", type=int, default=3, help="Apply loop penalty when revisiting same cell at or above this count")
    p.add_argument("--loop-visit-penalty", type=float, default=-0.10, help="Penalty for revisiting same cell too often")
    p.add_argument("--food-reward", type=float, default=10.0, help="Reward for eating food")
    p.add_argument("--death-penalty", type=float, default=-10.0, help="Penalty for wall/self collision")
    p.add_argument("--distance-reward-toward", type=float, default=0.1, help="Reward when move reduces Manhattan distance to food")
    p.add_argument("--distance-penalty-away", type=float, default=-0.05, help="Penalty when move increases Manhattan distance to food")
    p.add_argument("--idle-step-coeff", type=float, default=0.001, help="Linear hunger penalty coefficient: -coeff * steps_since_food")
    p.add_argument("--hunger-exp-base", type=float, default=0.0, help="Base coefficient for exponential hunger penalty")
    p.add_argument("--hunger-exp-gamma", type=float, default=1.005, help="Growth factor (>1) for exponential hunger penalty")
    p.add_argument("--hunger-exp-max", type=float, default=1.0, help="Upper cap for exponential hunger penalty")
    p.add_argument("--step-limit-penalty", type=float, default=0.0, help="Penalty when episode ends by step limit")

    p.add_argument("--save-every", type=int, default=50, help="Save checkpoint every N episodes")
    p.add_argument("--log-every", type=int, default=10, help="Print training log every N episodes")
    p.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="Output directory where ./checkpoints and ./logs are written",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cpu, cuda, cuda:0, ...",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = TrainConfig(
        episodes=args.episodes,
        grid_size=args.grid_size,
        max_steps_per_episode=args.max_steps_per_episode,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        warmup_steps=args.warmup_steps,
        target_update_steps=args.target_update_steps,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay_steps,
        max_steps_factor=args.max_steps_factor,
        loop_visit_threshold=args.loop_visit_threshold,
        loop_visit_penalty=args.loop_visit_penalty,
        food_reward=args.food_reward,
        death_penalty=args.death_penalty,
        distance_reward_toward=args.distance_reward_toward,
        distance_penalty_away=args.distance_penalty_away,
        idle_step_coeff=args.idle_step_coeff,
        hunger_exp_base=args.hunger_exp_base,
        hunger_exp_gamma=args.hunger_exp_gamma,
        hunger_exp_max=args.hunger_exp_max,
        step_limit_penalty=args.step_limit_penalty,
        save_every=args.save_every,
        log_every=args.log_every,
        out_dir=args.out_dir,
        seed=args.seed,
        device=args.device,
    )
    train(cfg)


if __name__ == "__main__":
    main()
