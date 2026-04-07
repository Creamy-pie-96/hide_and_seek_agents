"""
play.py — Playback trained policies or saved replays.

Usage:
    python play.py --sim-backend sim2 --episodes 5
    python play.py --sim-backend legacy --load checkpoints/final.pt --renderer pygame
    python play.py --sim-backend sim2 --replay outputs/replays/<run_id>/iter_000050_ep_01.json
"""

import argparse, sys, os, random
import json
import time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="Watch hide & seek agents play")
    p.add_argument("--sim-backend", type=str, default="sim2", choices=["sim2", "legacy"],
                   help="simulation backend for playback")
    p.add_argument("--load",     type=str,  default=None)
    p.add_argument("--replay",   type=str,  default=None,
                   help="path to saved replay JSON")
    p.add_argument("--episodes", type=int,  default=10)
    p.add_argument("--fps",      type=int,  default=12)
    p.add_argument("--width",    type=int,  default=36)
    p.add_argument("--height",   type=int,  default=36)
    p.add_argument("--device",   type=str,  default="cpu")
    p.add_argument("--renderer", type=str, default="pygame", choices=["pygame"])
    p.add_argument("--sim2-renderer", type=str, default="none", choices=["none"])
    p.add_argument("--hold-seconds", type=float, default=10.0,
                   help="seconds to keep renderer open after replay playback")
    p.add_argument("--asset-dir", type=str, default="assest",
                   help="directory containing 3D models (hider.fbx, seeker.fbx)")
    p.add_argument("--use-fbx-models", action="store_true",
                   help="use assest/hider.fbx and assest/seeker.fbx (off by default)")
    return p.parse_args()


def _build_renderer(args):
    if args.renderer == "pygame":
        from render.pygame_render import HideSeekRenderer
        tile_px = max(10, 600 // max(args.width, args.height))
        renderer = HideSeekRenderer(args.height, args.width, tile_px)
    else:
        from render.renderer_ursina import Ursina3DRenderer, UrsinaRendererConfig
        renderer = Ursina3DRenderer(
            args.height,
            args.width,
            config=UrsinaRendererConfig(asset_dir=args.asset_dir, use_fbx_models=args.use_fbx_models),
        )
    renderer.init()
    return renderer


def _build_sim2_renderer(args):
    return None


def _play_replay(args) -> None:
    renderer = _build_renderer(args)
    with open(args.replay, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        renderer.close()
        raise ValueError("Invalid replay format: top-level JSON must be an object")

    schema_version = payload.get("schema_version", 0)
    if schema_version not in (0, 1):
        renderer.close()
        raise ValueError(f"Unsupported replay schema_version: {schema_version}")

    frames = payload.get("frames", [])
    if not isinstance(frames, list) or len(frames) == 0:
        renderer.close()
        raise ValueError("Replay contains no frames")

    # B2 fix: validate each frame has required keys before playback.
    _REQUIRED_FRAME_KEYS = {"grid", "agents"}
    for idx, frame in enumerate(frames):
        if not isinstance(frame, dict):
            renderer.close()
            raise ValueError(f"Frame {idx}: expected dict, got {type(frame).__name__}")
        missing = _REQUIRED_FRAME_KEYS - frame.keys()
        if missing:
            renderer.close()
            raise ValueError(f"Frame {idx}: missing required keys: {sorted(missing)}")
        if not isinstance(frame["grid"], list) or len(frame["grid"]) == 0:
            renderer.close()
            raise ValueError(f"Frame {idx}: 'grid' must be a non-empty list")
        if not isinstance(frame["agents"], list):
            renderer.close()
            raise ValueError(f"Frame {idx}: 'agents' must be a list")

    print(f"Loaded replay with {len(frames)} frames: {args.replay}")
    for frame in frames:
        renderer.draw(frame, fps=args.fps)
    if args.hold_seconds > 0:
        print(f"Replay finished. Holding final frame for {args.hold_seconds:.1f}s...")
        time.sleep(args.hold_seconds)
    renderer.close()


def _play_replay_sim2(args) -> None:
    renderer = _build_sim2_renderer(args)
    with open(args.replay, "r", encoding="utf-8") as f:
        payload = json.load(f)

    frames = payload.get("frames", [])
    if not isinstance(frames, list) or not frames:
        if renderer is not None:
            renderer.close()
        raise ValueError("Replay contains no frames")

    print(f"Loaded sim2 replay with {len(frames)} frames: {args.replay}")
    if renderer is not None:
        for frame in frames:
            renderer.draw(frame, fps=args.fps)
        renderer.close()


def _play_sim2(args) -> None:
    from rl.sim2_runner import Sim2RolloutRunner, Sim2RunnerConfig

    runner = Sim2RolloutRunner(
        Sim2RunnerConfig(
            width=args.width,
            height=args.height,
            max_steps=400,
            prep_steps=20,
            output_root="outputs",
        )
    )

    renderer = _build_sim2_renderer(args)
    try:
        for ep in range(args.episodes):
            stats = runner.run_episode(
                seed=random.randint(0, 99999),
                renderer=renderer,
                fps=args.fps,
                record_video=False,
                record_replay=False,
                prefix=f"play_{ep+1:03d}",
            )
            print(
                f"[sim2-play] ep={ep+1} steps={stats['steps']} "
                f"hiders_alive={stats['alive_hiders']} seekers_alive={stats['alive_seekers']}"
            )
    finally:
        if renderer is not None:
            renderer.close()


def _play_legacy(args) -> None:
    from env.hide_seek_env import HideSeekEnv, HIDER_IDS, SEEKER_IDS
    from rl.checkpointing import safe_torch_load, validate_checkpoint_schema
    from rl.network import make_networks

    device = torch.device(args.device)

    env = HideSeekEnv(width=args.width, height=args.height)
    hider_net, seeker_net = make_networks(args.device)

    if args.load:
        ckpt = safe_torch_load(args.load, map_location=device)
        validate_checkpoint_schema(ckpt, ('hider_net', 'seeker_net'))
        hider_net.load_state_dict(ckpt['hider_net'])
        seeker_net.load_state_dict(ckpt['seeker_net'])
        print(f"Loaded checkpoint: {args.load}")
        use_trained = True
    else:
        print("No checkpoint — using random actions.")
        use_trained = False

    hider_net.eval()
    seeker_net.eval()

    renderer = _build_renderer(args)

    for ep in range(args.episodes):
        obs_dict, _ = env.reset(seed=random.randint(0, 99999))
        hidden = {i: None for i in range(6)}
        ep_rewards = {i: 0.0 for i in range(6)}
        running = True

        while running:
            action_dict = {}
            action_masks = env.get_action_masks()
            value_obs = env.build_global_state().astype(np.float32)
            with torch.no_grad():
                for agent in env.agents:
                    aid = agent.agent_id
                    if not agent.alive:
                        action_dict[aid] = 0
                        hidden[aid] = None
                        continue
                    if use_trained:
                        net = hider_net if aid in HIDER_IDS else seeker_net
                        obs_t = torch.tensor(obs_dict[aid], dtype=torch.float32,
                                             device=device).unsqueeze(0)
                        value_obs_t = torch.tensor(value_obs, dtype=torch.float32,
                                                   device=device).unsqueeze(0)
                        mask_t = torch.tensor(action_masks[aid], dtype=torch.float32,
                                              device=device).unsqueeze(0)
                        logits, _, new_h = net(obs_t, hidden[aid], value_obs=value_obs_t, action_mask=mask_t)
                        hidden[aid] = new_h
                        action_dict[aid] = logits.argmax(dim=-1).item()
                    else:
                        action_dict[aid] = random.randint(0, env.action_dim - 1)

            obs_dict, rewards, dones, _ = env.step(action_dict)
            for i, r in rewards.items():
                ep_rewards[i] += r

            running = renderer.draw(env.get_render_state(), fps=args.fps)
            if not running or dones['__all__']:
                break

        if not running:
            break

        h_ret = np.mean([ep_rewards[i] for i in HIDER_IDS])
        s_ret = np.mean([ep_rewards[i] for i in SEEKER_IDS])
        result = "HIDERS WIN" if env.hiders_caught < 3 else "SEEKERS WIN"
        print(f"Episode {ep+1}: {result} | H_ret={h_ret:+.0f}  S_ret={s_ret:+.0f} "
              f"| Caught={env.hiders_caught}/3")

    renderer.close()


def main():
    args   = parse_args()

    if args.replay:
        if args.sim_backend == "legacy":
            _play_replay(args)
        else:
            _play_replay_sim2(args)
        return

    if args.sim_backend == "legacy":
        print("[play] Running LEGACY backend (deprecated path).")
        _play_legacy(args)
    else:
        print("[play] Running SIM2 backend (default).")
        _play_sim2(args)


if __name__ == "__main__":
    main()