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
import random
from dataclasses import dataclass

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from env.hide_seek_env import HideSeekEnv
from rl.mappo           import MAPPOTrainer
from render.pygame_render import HideSeekRenderer
from render.renderer_3d import Viser3DRenderer


@dataclass(frozen=True)
class TrainConfig:
    rollouts: int
    no_render: bool
    render_every: int
    device: str
    load: str | None
    save_dir: str
    save_every: int
    width: int
    height: int
    seed: int | None
    renderer: str
    output_root: str
    run_id: str | None
    eval_every: int
    eval_episodes: int
    eval_fps: int
    no_eval_video: bool
    no_replay: bool
    tensorboard: bool


def set_global_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="Hide & Seek MAPPO Trainer")
    p.add_argument("--rollouts",    type=int, default=500,  help="training rollouts")
    p.add_argument("--no-render",   action="store_true",    help="disable visual rendering")
    p.add_argument("--renderer",    type=str, default="pygame", choices=["pygame", "viser3d"],
                   help="render backend when rendering is enabled")
    p.add_argument("--render-every",type=int, default=5,    help="render every N rollouts")
    p.add_argument("--device",      type=str, default="cpu",help="cpu or cuda")
    p.add_argument("--load",        type=str, default=None, help="checkpoint to load")
    p.add_argument("--save-dir",    type=str, default="checkpoints")
    p.add_argument("--save-every",  type=int, default=50)
    p.add_argument("--width",       type=int, default=36)
    p.add_argument("--height",      type=int, default=36)
    p.add_argument("--seed",        type=int, default=None)
    p.add_argument("--output-root", type=str, default="outputs")
    p.add_argument("--run-id",      type=str, default=None)
    p.add_argument("--eval-every",  type=int, default=25,
                   help="run deterministic evaluation every N rollouts (0 disables)")
    p.add_argument("--eval-episodes", type=int, default=1)
    p.add_argument("--eval-fps",    type=int, default=12)
    p.add_argument("--no-eval-video", action="store_true")
    p.add_argument("--no-replay",   action="store_true")
    p.add_argument("--tensorboard", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(**vars(args))
    set_global_seed(cfg.seed)

    env = HideSeekEnv(
        width    = cfg.width,
        height   = cfg.height,
        n_food   = 14,
        n_heavy  = 4,
        seed     = cfg.seed,
    )

    trainer = MAPPOTrainer(
        env          = env,
        device       = cfg.device,
        render       = not cfg.no_render,
        render_every = cfg.render_every,
    )

    if cfg.load:
        trainer.load(cfg.load)

    renderer = None
    if not cfg.no_render:
        if cfg.renderer == "pygame":
            tile_px  = max(10, 600 // max(cfg.width, cfg.height))
            renderer = HideSeekRenderer(
                grid_h  = cfg.height,
                grid_w  = cfg.width,
                tile_px = tile_px,
            )
        else:
            renderer = Viser3DRenderer(
                grid_h=cfg.height,
                grid_w=cfg.width,
            )
        renderer.init()
        print("Renderer started. Press ESC or close window to stop.")

    try:
        trainer.train(
            n_rollouts = cfg.rollouts,
            save_every = cfg.save_every,
            save_path  = cfg.save_dir,
            renderer   = renderer,
            output_root = cfg.output_root,
            run_id = cfg.run_id,
            eval_every = cfg.eval_every,
            eval_episodes = cfg.eval_episodes,
            eval_fps = cfg.eval_fps,
            save_eval_videos = not cfg.no_eval_video,
            save_replays = not cfg.no_replay,
            tensorboard = cfg.tensorboard,
        )
    finally:
        if renderer:
            renderer.close()


if __name__ == "__main__":
    main()