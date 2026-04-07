"""
train.py — Launch training with live Pygame visualization.

Usage:
    python train.py --sim-backend sim2 --rollouts 500
    python train.py --sim-backend sim2 --no-render
    python train.py --sim-backend legacy --load checkpoints/checkpoint_100.pt
    python train.py --sim-backend legacy --rollouts 1000 --device cuda
"""

import argparse
import sys
import os
import random
from dataclasses import dataclass

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))


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
    asset_dir: str
    use_fbx_models: bool
    sim_backend: str
    sim2_renderer: str
    curriculum: bool
    curriculum_min_rollouts_per_level: int
    curriculum_eval_window: int
    curriculum_promote_hiders_caught_mean: float
    curriculum_promote_seeker_return_mean: float


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
    p.add_argument("--sim-backend", type=str, default="sim2", choices=["sim2", "legacy"],
                   help="simulation backend: sim2 (new primitive simulator) or legacy (old env)")
    p.add_argument("--rollouts",    type=int, default=500,  help="training rollouts")
    p.add_argument("--no-render",   action="store_true",    help="disable visual rendering")
    p.add_argument("--renderer",    type=str, default="pygame", choices=["pygame"],
                   help="render backend when rendering is enabled")
    p.add_argument("--sim2-renderer", type=str, default="none", choices=["none"],
                   help="renderer for sim2 backend")
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
    p.add_argument("--asset-dir",   type=str, default="assest",
                   help="directory containing 3D models (hider.fbx, seeker.fbx)")
    p.add_argument("--use-fbx-models", action="store_true",
                   help="use assest/hider.fbx and assest/seeker.fbx (off by default)")
    p.add_argument("--curriculum", action="store_true",
                   help="enable legacy easy-to-hard curriculum progression")
    p.add_argument("--curriculum-min-rollouts-per-level", type=int, default=60,
                   help="minimum rollouts before a curriculum promotion is considered")
    p.add_argument("--curriculum-eval-window", type=int, default=20,
                   help="recent episodes window used for curriculum promotion metrics")
    p.add_argument("--curriculum-promote-hiders-caught-mean", type=float, default=1.2,
                   help="promote when mean hiders caught over eval window reaches this threshold")
    p.add_argument("--curriculum-promote-seeker-return-mean", type=float, default=-20.0,
                   help="promote when mean seeker return over eval window reaches this threshold")
    return p.parse_args()


def _run_legacy(cfg: TrainConfig) -> None:
    from env.hide_seek_env import HideSeekEnv
    from rl.mappo import MAPPOTrainer
    from rl.curriculum import LegacySeekerCurriculumManager

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

    curriculum_manager = None
    if cfg.curriculum:
        curriculum_manager = LegacySeekerCurriculumManager(
            base_seed=cfg.seed,
            min_rollouts_per_level=cfg.curriculum_min_rollouts_per_level,
            eval_window=cfg.curriculum_eval_window,
            promote_if_hiders_caught_mean_at_least=cfg.curriculum_promote_hiders_caught_mean,
            promote_if_seeker_return_mean_at_least=cfg.curriculum_promote_seeker_return_mean,
        )
        print("[train] Legacy curriculum enabled.")

    renderer = None
    if not cfg.no_render:
        if cfg.renderer == "pygame":
            from render.pygame_render import HideSeekRenderer
            tile_px  = max(10, 600 // max(cfg.width, cfg.height))
            renderer = HideSeekRenderer(
                grid_h  = cfg.height,
                grid_w  = cfg.width,
                tile_px = tile_px,
            )
        else:
            from render.renderer_ursina import Ursina3DRenderer, UrsinaRendererConfig
            renderer = Ursina3DRenderer(
                grid_h=cfg.height,
                grid_w=cfg.width,
                config=UrsinaRendererConfig(asset_dir=cfg.asset_dir, use_fbx_models=cfg.use_fbx_models),
            )
        renderer.init()
        print("Legacy renderer started. Press ESC or close window to stop.")

    try:
        trainer.train(
            n_rollouts = cfg.rollouts,
            save_every = cfg.save_every,
            save_path  = cfg.save_dir,
            renderer   = renderer,
            curriculum_manager = curriculum_manager,
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


def _run_sim2(cfg: TrainConfig) -> None:
    from rl.sim2_runner import Sim2RolloutRunner, Sim2RunnerConfig

    runner = Sim2RolloutRunner(
        Sim2RunnerConfig(
            width=cfg.width,
            height=cfg.height,
            max_steps=400,
            prep_steps=20,
            output_root=cfg.output_root,
            run_id=cfg.run_id,
        )
    )

    renderer = None

    try:
        for i in range(cfg.rollouts):
            ep_seed = (cfg.seed or 0) + i
            stats = runner.run_episode(
                seed=ep_seed,
                renderer=renderer,
                fps=cfg.eval_fps,
                record_video=not cfg.no_eval_video,
                record_replay=not cfg.no_replay,
                prefix=f"iter_{i+1:06d}",
            )
            print(
                f"[sim2-train] iter={i+1}/{cfg.rollouts} steps={stats['steps']} "
                f"hiders_alive={stats['alive_hiders']} seekers_alive={stats['alive_seekers']}"
            )
    finally:
        if renderer is not None:
            renderer.close()


def main():
    args = parse_args()
    cfg = TrainConfig(**vars(args))
    set_global_seed(cfg.seed)
    if cfg.sim_backend == "legacy":
        print("[train] Running LEGACY backend (deprecated path).")
        _run_legacy(cfg)
    else:
        print("[train] Running SIM2 backend (default).")
        _run_sim2(cfg)


if __name__ == "__main__":
    main()