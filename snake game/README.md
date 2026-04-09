# Snake RL (CNN-PPO)

This folder is a standalone sub-project for training a Snake agent with a CNN-based PPO policy.

## Features

- 3-channel spatial observation `(20, 20, 3)`:
  - head mask
  - body mask
  - food mask
- Relative action space:
  - `0`: straight
  - `1`: right turn
  - `2`: left turn
- PPO actor-critic with GAE($\lambda$), clipping, entropy regularization.
- Vectorized environment rollout collection.
- Reward shaping + progressive hunger penalties.
- Deterministic evaluation loop and replay export for first/best/last eval episodes.

## Structure

- [snake_game/env.py](snake_game/env.py): Snake environment + pygame render
- [snake_game/ppo_model.py](snake_game/ppo_model.py): CNN actor-critic model
- [snake_game/ppo_agent.py](snake_game/ppo_agent.py): PPO optimization logic
- [snake_game/ppo_train.py](snake_game/ppo_train.py): PPO training + eval + replay export
- [snake_game/ppo_play.py](snake_game/ppo_play.py): PPO inference/play loop
- [train.py](train.py): wrapper entrypoint
- [play.py](play.py): wrapper entrypoint

## Run

From this folder ([snake game](.)):

- Train:
  - `/home/DATA/CODE/code/hide_and_seek/.venv/bin/python train.py --episodes 3000 --device cuda`
- Play:
  - `/home/DATA/CODE/code/hide_and_seek/.venv/bin/python play.py --checkpoint ./checkpoints/final.pt`

## Built-in CLI help

Both programs include full `--help` output with examples and flag descriptions.

- Train help:
  - `/home/DATA/CODE/code/hide_and_seek/.venv/bin/python train.py --help`
- Play help:
  - `/home/DATA/CODE/code/hide_and_seek/.venv/bin/python play.py --help`

## Training flags (`train.py`)

PPO training now includes:

- core PPO args (`--n-envs`, `--rollout-steps`, `--update-epochs`, `--clip-coef`, `--ent-coef`, etc.)
- reward shaping args (`--distance-reward-toward`, `--distance-penalty-away`, `--idle-step-coeff`, etc.)
- evaluation args (`--eval-every`, `--eval-episodes`)
- resume args (`--resume`, `--resume-reset-optim`)
- adaptive entropy controller args (`--use-adaptive-entropy`, `--entropy-min`, `--entropy-max`, ...)
- proficiency curriculum args (`--use-curriculum`, `--curriculum-promote-streak`)
- static opponent/self-play args (`--self-play`, `--self-play-mode heuristic|last_best`, `--opponent-food-penalty`)

Use built-in help for the full list:

- `/home/DATA/CODE/code/hide_and_seek/.venv/bin/python train.py --help`

## Play flags (`play.py`)

| Flag           | Default                  | What it does                                          |
| :------------- | :----------------------- | :---------------------------------------------------- |
| `--checkpoint` | `./checkpoints/final.pt` | Path to model checkpoint file to load.                |
| `--episodes`   | `3`                      | Number of gameplay episodes to run.                   |
| `--grid-size`  | `20`                     | Board size. Should match the model you trained.       |
| `--fps`        | `12`                     | Render speed (frames per second).                     |
| `--max-steps`  | `4000`                   | Max steps per play episode.                           |
| `--device`     | `auto`                   | Compute device: `auto`, `cpu`, `cuda`, `cuda:0`, etc. |
| `--stochastic` | `False`                  | Sample action from policy instead of greedy action.   |

## Quick smoke test

`cd "/home/DATA/CODE/code/hide_and_seek/snake game" && /home/DATA/CODE/code/hide_and_seek/.venv/bin/python train.py --episodes 40 --n-envs 4 --rollout-steps 128 --update-epochs 2 --minibatch-size 256 --eval-every 20 --eval-episodes 4 --save-every 40 --log-every 10 --device cuda`

## Resume training example

`cd "/home/DATA/CODE/code/hide_and_seek/snake game" && /home/DATA/CODE/code/hide_and_seek/.venv/bin/python train.py --episodes 5000 --resume checkpoints/final.pt --self-play --self-play-mode last_best`
