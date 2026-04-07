"""Headless-first sim2 runner with optional primitive rendering."""

from __future__ import annotations

import argparse

from rl.sim2_runner import Sim2RolloutRunner, Sim2RunnerConfig


def parse_args():
    p = argparse.ArgumentParser(description="Run sim2 episodes with optional renderer")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--width", type=int, default=24)
    p.add_argument("--height", type=int, default=24)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--prep-steps", type=int, default=20)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--output-root", type=str, default="outputs")
    p.add_argument("--renderer", type=str, default="none", choices=["none"])
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--no-replay", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runner = Sim2RolloutRunner(
        Sim2RunnerConfig(
            width=args.width,
            height=args.height,
            max_steps=args.max_steps,
            prep_steps=args.prep_steps,
            output_root=args.output_root,
        )
    )

    renderer = None

    try:
        for i in range(args.episodes):
            ep_seed = args.seed + i
            stats = runner.run_episode(
                seed=ep_seed,
                renderer=renderer,
                fps=args.fps,
                record_video=not args.no_video,
                record_replay=not args.no_replay,
                prefix=f"iter_{i+1:03d}",
            )
            print(
                f"[sim2] ep={i+1} seed={ep_seed} steps={stats['steps']} "
                f"hiders_alive={stats['alive_hiders']} seekers_alive={stats['alive_seekers']} "
                f"video={stats['video_path']} replay={stats['replay_path']}"
            )
    finally:
        if renderer is not None:
            renderer.close()


if __name__ == "__main__":
    main()
