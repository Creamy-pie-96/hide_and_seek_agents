"""Simple runner for sim2 with optional primitive Ursina viewer."""

from __future__ import annotations

import argparse
import random

from sim2.core import PrimitiveHideSeekSim
from sim2.entities import Action


def parse_args():
    p = argparse.ArgumentParser(description="Play sim2 primitive hide-and-seek")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--width", type=int, default=24)
    p.add_argument("--height", type=int, default=24)
    p.add_argument("--renderer", type=str, default="none", choices=["none"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sim = PrimitiveHideSeekSim(width=args.width, height=args.height)
    sim.reset(seed=args.seed)

    renderer = None

    try:
        for _ in range(args.steps):
            actions = {i: random.randint(int(Action.STAY), int(Action.RIGHT)) for i in range(6)}
            state, rewards, done, info = sim.step(actions)
            if renderer is not None:
                alive = renderer.draw(state, fps=args.fps)
                if not alive:
                    break
            if done["__all__"]:
                sim.reset(seed=args.seed)
    finally:
        if renderer is not None:
            renderer.close()


if __name__ == "__main__":
    main()
