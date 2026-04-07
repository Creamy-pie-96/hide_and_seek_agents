# Snake RL (CNN-DQN)

This folder is a standalone sub-project for training a Snake agent with a CNN-based DQN.

## Features

- 3-channel spatial observation `(20, 20, 3)`:
  - head mask
  - body mask
  - food mask
- Relative action space:
  - `0`: straight
  - `1`: right turn
  - `2`: left turn
- Reward shaping:
  - eat food: `+10`
  - collision: `-10`
  - move toward food: `+0.1`
  - move away from food: `-0.1`
  - step penalty: `-0.01`
  - loop/stall penalty: `-0.5`
- Efficient replay buffer with uint8 storage.
- Double DQN target computation + periodic target updates.

## Structure

- [snake_game/env.py](snake_game/env.py): Snake environment + pygame render
- [snake_game/model.py](snake_game/model.py): CNN Q-network
- [snake_game/replay.py](snake_game/replay.py): replay buffer
- [snake_game/agent.py](snake_game/agent.py): DQN agent logic
- [snake_game/train.py](snake_game/train.py): training entry logic
- [snake_game/play.py](snake_game/play.py): inference/play loop
- [train.py](train.py): wrapper entrypoint
- [play.py](play.py): wrapper entrypoint

## Run

From this folder ([snake game](.)):

- Train:
  - `/home/DATA/CODE/code/hide_and_seek/.venv/bin/python train.py --episodes 1500`
- Play:
  - `/home/DATA/CODE/code/hide_and_seek/.venv/bin/python play.py --checkpoint ./checkpoints/final.pt`

## Built-in CLI help

Both programs include full `--help` output with examples and flag descriptions.

- Train help:
  - `/home/DATA/CODE/code/hide_and_seek/.venv/bin/python train.py --help`
- Play help:
  - `/home/DATA/CODE/code/hide_and_seek/.venv/bin/python play.py --help`

## Training flags (`train.py`)

| Flag                      | Default  | What it does                                                |
| :------------------------ | :------- | :---------------------------------------------------------- |
| `--episodes`              | `1500`   | Number of episodes to train.                                |
| `--grid-size`             | `20`     | Board size $N$ for an $N \times N$ grid.                    |
| `--max-steps-per-episode` | `4000`   | Max steps allowed in one episode.                           |
| `--lr`                    | `1e-4`   | Adam learning rate.                                         |
| `--gamma`                 | `0.99`   | Discount factor for future rewards.                         |
| `--batch-size`            | `64`     | Minibatch size sampled from replay buffer.                  |
| `--replay-capacity`       | `100000` | Replay buffer maximum transitions.                          |
| `--warmup-steps`          | `2000`   | Minimum stored transitions before optimization starts.      |
| `--target-update-steps`   | `1000`   | How often to copy online network weights to target network. |
| `--eps-start`             | `1.0`    | Initial epsilon for epsilon-greedy exploration.             |
| `--eps-end`               | `0.01`   | Final epsilon floor after decay.                            |
| `--eps-decay-steps`       | `200000` | Environment steps over which epsilon decays linearly.       |
| `--save-every`            | `50`     | Save checkpoint every N episodes.                           |
| `--log-every`             | `10`     | Print training logs every N episodes.                       |
| `--out-dir`               | `.`      | Output root; writes `checkpoints/` and `logs/` inside it.   |
| `--seed`                  | `42`     | Random seed for reproducibility.                            |
| `--device`                | `auto`   | Compute device: `auto`, `cpu`, `cuda`, `cuda:0`, etc.       |

## Play flags (`play.py`)

| Flag           | Default                  | What it does                                          |
| :------------- | :----------------------- | :---------------------------------------------------- |
| `--checkpoint` | `./checkpoints/final.pt` | Path to model checkpoint file to load.                |
| `--episodes`   | `3`                      | Number of gameplay episodes to run.                   |
| `--grid-size`  | `20`                     | Board size. Should match the model you trained.       |
| `--fps`        | `12`                     | Render speed (frames per second).                     |
| `--max-steps`  | `4000`                   | Max steps per play episode.                           |
| `--device`     | `auto`                   | Compute device: `auto`, `cpu`, `cuda`, `cuda:0`, etc. |

## Quick smoke test

- `/home/DATA/CODE/code/hide_and_seek/.venv/bin/python train.py --episodes 5 --max-steps-per-episode 200 --warmup-steps 64 --save-every 5 --log-every 1`
