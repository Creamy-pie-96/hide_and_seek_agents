from __future__ import annotations

import argparse
import os
import time

import torch

from .agent import AgentConfig, DQNAgent
from .env import SnakeEnv


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def run(
    checkpoint: str,
    episodes: int = 3,
    grid_size: int = 20,
    fps: int = 12,
    max_steps: int = 4000,
    device: str = "auto",
) -> None:
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)

    env = SnakeEnv(grid_size=grid_size)
    agent = DQNAgent(grid_size=grid_size, device=dev, cfg=AgentConfig())

    if checkpoint and os.path.exists(checkpoint):
        agent.load(checkpoint, strict=False)
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print("Checkpoint not found. Running with random-initialized network.")

    for ep in range(1, episodes + 1):
        s = env.reset()
        total_r = 0.0
        for t in range(max_steps):
            env.render(fps=fps)
            a = agent.act(s, epsilon=0.0)
            s, r, done, truncated, info = env.step(a)
            total_r += r
            if done or truncated:
                print(
                    f"[play ep {ep}] steps={t+1} score={info['score']} "
                    f"length={info['length']} reward={total_r:.2f} reason={info['reason']}"
                )
                break

        # brief pause between episodes for readability
        time.sleep(0.3)

    env.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Play Snake using a trained DQN checkpoint",
        formatter_class=_HelpFormatter,
        epilog=(
            "Examples:\n"
            "  python play.py --checkpoint ./checkpoints/final.pt\n"
            "  python play.py --checkpoint ./checkpoints/best.pt --episodes 5 --fps 20"
        ),
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/final.pt",
        help="Path to a trained checkpoint (.pt)",
    )
    p.add_argument("--episodes", type=int, default=3, help="How many episodes to run")
    p.add_argument(
        "--grid-size",
        type=int,
        default=20,
        help="Board size N for an N x N grid (must match trained model)",
    )
    p.add_argument("--fps", type=int, default=12, help="Render frames per second")
    p.add_argument("--max-steps", type=int, default=4000, help="Maximum steps per episode")
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cpu, cuda, cuda:0, ...",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        grid_size=args.grid_size,
        fps=args.fps,
        max_steps=args.max_steps,
        device=args.device,
    )


if __name__ == "__main__":
    main()
