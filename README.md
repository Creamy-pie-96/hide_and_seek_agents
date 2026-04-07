# hide_and_seek_agents

Multi-agent hide-and-seek project with a model-free `sim2` backend (default) and a legacy MAPPO backend retained for migration.

## Quick start

1. Create/activate virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train (default `sim2` backend):

```bash
python train.py --rollouts 500 --sim-backend sim2 --sim2-renderer none
```

Headless `sim2` run:

```bash
python train.py --rollouts 500 --sim-backend sim2 --no-render
```

Legacy MAPPO training path (deprecated):

```bash
python train.py --sim-backend legacy --rollouts 500 --device cpu
```

Legacy MAPPO on GPU:

```bash
python train.py --sim-backend legacy --rollouts 500 --device cuda
```

4. Watch play (default `sim2`):

```bash
python play.py --sim-backend sim2 --episodes 5 --sim2-renderer none
```

Legacy policy play:

```bash
python play.py --sim-backend legacy --load checkpoints/final.pt --episodes 5
```

Replay playback:

```bash
python play.py --sim-backend sim2 --replay outputs/replays/<run_id>/iter_000025_ep_01.json
```

## Command help

Both entrypoints already support `--help` via argparse:

```bash
python train.py --help
python play.py --help
```

## Train flags

- `--sim-backend {sim2,legacy}`: select simulator backend.
- `--rollouts N`: number of training iterations/episodes.
- `--no-render`: run headless (recommended for faster runs).
- `--device {cpu,cuda}`: device for legacy MAPPO backend.
- `--output-root PATH`: where logs/replays/videos/tensorboard are stored.
- `--run-id ID`: explicit run id for artifact folders.
- `--no-eval-video`: disable MP4 generation.
- `--no-replay`: disable replay JSON generation.
- `--tensorboard`: enable TensorBoard logging (legacy backend).

## Play flags

- `--sim-backend {sim2,legacy}`: choose backend for playback.
- `--episodes N`: number of episodes to run.
- `--replay PATH`: play a saved replay JSON.
- `--load PATH`: load legacy model checkpoint and run policy playback.
- `--fps N`: playback FPS.
- `--sim2-renderer none`: sim2 playback mode (headless-compatible path).

## Saved outputs and checkpoints

Training writes artifacts under:

- `outputs/logs/<run_id>/metrics.csv`
- `outputs/eval_videos/<run_id>/*.mp4`
- `outputs/replays/<run_id>/*.json`
- `outputs/tensorboard/<run_id>/` (legacy with `--tensorboard`)
- `checkpoints/final.pt` (legacy MAPPO model)

These are kept so you can replay and inspect what the model learned later.

## Project layout

- [env/world.py](env/world.py): procedural map, room/light graph, mutable world state
- [env/objects.py](env/objects.py): typed object state and lifecycle helpers
- [env/agent.py](env/agent.py): agent state, action execution, observation generation
- [env/hide_seek_env.py](env/hide_seek_env.py): custom parallel multi-agent environment API
- [rl/network.py](rl/network.py): shared policy network + centralized critic path
- [rl/memory.py](rl/memory.py): rollout memory and tensor conversion helpers
- [rl/mappo.py](rl/mappo.py): trainer and PPO update loop
- [render/pygame_render.py](render/pygame_render.py): visualization
- [sim2/core.py](sim2/core.py): model-free simulator core
- [rl/sim2_runner.py](rl/sim2_runner.py): headless-first rollout runner + artifact generation

## Mechanics currently implemented

- 3 hiders vs 3 seekers
- Procedural BSP room generation
- Food, fake food, scent, barricades, heavy objects, light switches
- Team mechanics:
  - heavy push (hiders)
  - blackout trigger (3 hiders, distinct switched rooms)
  - coordinated sweep from distinct doors (seekers)
  - relay vision (adjacent seekers)
- Light-off reveal ping (risk mechanic)
- Action masking for invalid operations
- Hider anti-stagnation shaping (anti-lazy policy pressure)
- Agent identity one-hot in observations (symmetry breaking)
- Global-state centralized critic input
- Sequence-aware recurrent PPO updates
- Policy-pool self-play against historical opponents

## Validation

Run compile + tests:

```bash
python -m py_compile env/world.py env/objects.py env/agent.py env/hide_seek_env.py rl/network.py rl/memory.py rl/mappo.py render/pygame_render.py train.py play.py
python -m unittest discover -s tests -p 'test_*.py'
```

## Monitoring outputs

Training now writes experiment artifacts under:

- outputs/logs/<run_id>/metrics.csv
- outputs/eval_videos/<run_id>/<iteration>.mp4
- outputs/replays/<run*id>/iter*<iteration>_ep_<episode>.json
- outputs/tensorboard/<run_id>/

Launch TensorBoard:

```bash
tensorboard --logdir outputs/tensorboard
```

## Notes

- Checkpoints are expected to be trusted project-generated files.
- API is custom dict-based multi-agent (not strict PettingZoo).

Common commands:

```bash
python train.py --sim-backend sim2 --rollouts 100 --no-render
python train.py --sim-backend legacy --rollouts 100 --device cuda
python play.py --sim-backend sim2 --replay outputs/replays/<run_id>/<file>.json
```
