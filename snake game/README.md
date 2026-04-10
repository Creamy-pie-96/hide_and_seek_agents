# Snake CLI API Reference

This README is a flag reference for running play/train commands.

Difficulty levels (curriculum):

| Level | Mode                                                    |
| ----- | ------------------------------------------------------- |
| 0     | solo static food                                        |
| 1     | solo static blocks                                      |
| 2     | solo moving blocks                                      |
| 3     | solo moving blocks + moving food                        |
| 4     | competitive (moving obstacles + moving food + opponent) |

## Play command

Base command:

`/home/DATA/CODE/code/hide_and_seek/.venv/bin/python play.py [flags]`

| Flag                                         |                  Default | Description                                                  |
| -------------------------------------------- | -----------------------: | ------------------------------------------------------------ |
| `--checkpoint`                               | `./checkpoints/final.pt` | Model checkpoint path                                        |
| `--episodes`                                 |                      `3` | Number of episodes                                           |
| `--grid-size`                                |                     `20` | Board size                                                   |
| `--fps`                                      |                     `12` | Render FPS                                                   |
| `--max-steps`                                |                   `4000` | Max steps per episode                                        |
| `--device`                                   |                   `auto` | `auto`, `cpu`, `cuda`, `cuda:0`, ...                         |
| `--stochastic`                               |                  `False` | Sample actions instead of greedy                             |
| `--opponent-mode`                            |                   `none` | `none` or `heuristic`                                        |
| `--opponent-food-penalty`                    |                  `-0.10` | Reward penalty when opponent eats                            |
| `--opponent-random-prob`                     |                   `0.80` | Opponent randomness                                          |
| `--obstacle-count`                           |                      `0` | Number of blocks                                             |
| `--moving-obstacles / --no-moving-obstacles` |                  `False` | Toggle moving obstacles                                      |
| `--obstacle-move-period`                     |                      `8` | Obstacle movement period                                     |
| `--moving-food / --no-moving-food`           |                  `False` | Toggle moving food                                           |
| `--food-move-prob`                           |                   `0.15` | Food movement probability                                    |
| `--terminal-win-reward`                      |                    `3.0` | Terminal win bonus                                           |
| `--terminal-loss-penalty`                    |                   `-3.0` | Terminal loss penalty                                        |
| `--starvation-steps-factor`                  |                     `60` | Starvation limit factor                                      |
| `--starvation-penalty`                       |                   `-6.0` | Starvation penalty                                           |
| `--wall-follow-threshold`                    |                      `6` | Edge-follow threshold                                        |
| `--wall-follow-penalty`                      |                  `-0.12` | Edge-follow penalty                                          |
| `--curriculum`                               |                  `False` | Play all levels from 0 to 4; runs `--episodes` at each level |

Examples:

- Single mode:
  - `/home/DATA/CODE/code/hide_and_seek/.venv/bin/python play.py --checkpoint ./checkpoints/final.pt --episodes 3`
- Curriculum mode (3 episodes per level):
  - `/home/DATA/CODE/code/hide_and_seek/.venv/bin/python play.py --checkpoint ./checkpoints/final.pt --curriculum --episodes 3`

## Train command

Base command:

`/home/DATA/CODE/code/hide_and_seek/.venv/bin/python train.py [flags]`

| Flag                                                           |  Default | Description                            |
| -------------------------------------------------------------- | -------: | -------------------------------------- |
| `--episodes`                                                   |   `3000` | Total episodes target                  |
| `--grid-size`                                                  |     `20` | Final board size                       |
| `--seed`                                                       |     `42` | Random seed                            |
| `--device`                                                     |   `auto` | `auto`, `cpu`, `cuda`, ...             |
| `--lr`                                                         | `0.0004` | Learning rate                          |
| `--n-envs`                                                     |     `12` | Number of vectorized envs              |
| `--rollout-steps`                                              |    `256` | PPO rollout horizon                    |
| `--update-epochs`                                              |      `4` | PPO epochs per update                  |
| `--minibatch-size`                                             |    `512` | PPO minibatch size                     |
| `--eval-every`                                                 |     `25` | Evaluate every N episodes              |
| `--eval-episodes`                                              |     `10` | Eval episodes per eval call            |
| `--resume`                                                     |     `""` | Resume from checkpoint                 |
| `--use-curriculum / --no-use-curriculum`                       |   `True` | Difficulty curriculum                  |
| `--use-grid-curriculum / --no-use-grid-curriculum`             |   `True` | 10x10 to 20x20 progression             |
| `--grid-size-start`                                            |     `10` | Start grid size for grid curriculum    |
| `--grid-curriculum-min-episodes`                               |    `250` | Min episodes before grid promotion     |
| `--grid-curriculum-promote-score`                              |   `0.90` | Eval score gate for grid promotion     |
| `--grid-curriculum-max-starvation`                             |   `0.55` | Max starvation gate for grid promotion |
| `--food-centered-observation / --no-food-centered-observation` |   `True` | Enable centered observations           |

Use help for full list:

- `/home/DATA/CODE/code/hide_and_seek/.venv/bin/python train.py --help`
- `/home/DATA/CODE/code/hide_and_seek/.venv/bin/python play.py --help`
