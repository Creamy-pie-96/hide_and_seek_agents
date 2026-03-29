"""
train.py — Launch training with live Pygame visualization.

Usage:
    python train.py                   # train from scratch with renderer
    python train.py --no-render       # headless (faster)
    python train.py --load checkpoints/checkpoint_100.pt
    python train.py --rollouts 1000 --device cuda
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from env.hide_seek_env import HideSeekEnv
from rl.mappo           import MAPPOTrainer
from render.pygame_render import HideSeekRenderer


def parse_args():
    p = argparse.ArgumentParser(description="Hide & Seek MAPPO Trainer")
    p.add_argument("--rollouts",    type=int, default=500,  help="training rollouts")
    p.add_argument("--no-render",   action="store_true",    help="disable pygame")
    p.add_argument("--render-every",type=int, default=5,    help="render every N rollouts")
    p.add_argument("--device",      type=str, default="cpu",help="cpu or cuda")
    p.add_argument("--load",        type=str, default=None, help="checkpoint to load")
    p.add_argument("--save-dir",    type=str, default="checkpoints")
    p.add_argument("--save-every",  type=int, default=50)
    p.add_argument("--width",       type=int, default=36)
    p.add_argument("--height",      type=int, default=36)
    p.add_argument("--seed",        type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    env = HideSeekEnv(
        width    = args.width,
        height   = args.height,
        n_food   = 14,
        n_heavy  = 4,
        seed     = args.seed,
    )

    trainer = MAPPOTrainer(
        env          = env,
        device       = args.device,
        render       = not args.no_render,
        render_every = args.render_every,
    )

    if args.load:
        trainer.load(args.load)

    renderer = None
    if not args.no_render:
        tile_px  = max(10, 600 // max(args.width, args.height))
        renderer = HideSeekRenderer(
            grid_h  = args.height,
            grid_w  = args.width,
            tile_px = tile_px,
        )
        renderer.init()
        print("Renderer started. Press ESC or close window to stop.")

    try:
        trainer.train(
            n_rollouts = args.rollouts,
            save_every = args.save_every,
            save_path  = args.save_dir,
            renderer   = renderer,
        )
    finally:
        if renderer:
            renderer.close()


if __name__ == "__main__":
    main()