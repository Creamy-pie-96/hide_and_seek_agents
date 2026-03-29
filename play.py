"""
play.py — Watch trained agents play hide and seek.

Usage:
    python play.py                              # random agents (no checkpoint)
    python play.py --load checkpoints/final.pt  # trained agents
    python play.py --load checkpoints/final.pt --episodes 5 --fps 8
"""

import argparse, sys, os, random
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from env.hide_seek_env    import HideSeekEnv, HIDER_IDS, SEEKER_IDS
from rl.network           import make_networks
from render.pygame_render import HideSeekRenderer


def parse_args():
    p = argparse.ArgumentParser(description="Watch hide & seek agents play")
    p.add_argument("--load",     type=str,  default=None)
    p.add_argument("--episodes", type=int,  default=10)
    p.add_argument("--fps",      type=int,  default=12)
    p.add_argument("--width",    type=int,  default=36)
    p.add_argument("--height",   type=int,  default=36)
    p.add_argument("--device",   type=str,  default="cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    env = HideSeekEnv(width=args.width, height=args.height)
    hider_net, seeker_net = make_networks(args.device)

    if args.load:
        ckpt = torch.load(args.load, map_location=device)
        if not isinstance(ckpt, dict) or 'hider_net' not in ckpt or 'seeker_net' not in ckpt:
            raise ValueError("Invalid checkpoint format")
        hider_net.load_state_dict(ckpt['hider_net'])
        seeker_net.load_state_dict(ckpt['seeker_net'])
        print(f"Loaded checkpoint: {args.load}")
        use_trained = True
    else:
        print("No checkpoint — using random actions.")
        use_trained = False

    hider_net.eval()
    seeker_net.eval()

    tile_px  = max(10, 600 // max(args.width, args.height))
    renderer = HideSeekRenderer(args.height, args.width, tile_px)
    renderer.init()

    for ep in range(args.episodes):
        obs_dict, _ = env.reset(seed=random.randint(0, 99999))
        hidden = {i: None for i in range(6)}
        ep_rewards = {i: 0.0 for i in range(6)}
        running = True

        while running:
            action_dict = {}
            action_masks = env.get_action_masks()
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
                        mask_t = torch.tensor(action_masks[aid], dtype=torch.float32,
                                              device=device).unsqueeze(0)
                        logits, _, new_h = net(obs_t, hidden[aid], action_mask=mask_t)
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


if __name__ == "__main__":
    main()