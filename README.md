# hide_and_seek_agents

Multi-agent hide-and-seek environment and MAPPO trainer (3v3) built with Python, PyTorch, and Pygame.

## Quick start

1. Create/activate virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train:

```bash
python train.py --rollouts 500 --device cpu
```

4. Watch play:

```bash
python play.py --load checkpoints/final.pt --episodes 5
```

## Project layout

- [env/world.py](env/world.py): procedural map, room/light graph, mutable world state
- [env/objects.py](env/objects.py): typed object state and lifecycle helpers
- [env/agent.py](env/agent.py): agent state, action execution, observation generation
- [env/hide_seek_env.py](env/hide_seek_env.py): custom parallel multi-agent environment API
- [rl/network.py](rl/network.py): shared policy network + centralized critic path
- [rl/memory.py](rl/memory.py): rollout memory and tensor conversion helpers
- [rl/mappo.py](rl/mappo.py): trainer and PPO update loop
- [render/pygame_render.py](render/pygame_render.py): visualization

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

## Notes

- Checkpoints are expected to be trusted project-generated files.
- API is custom dict-based multi-agent (not strict PettingZoo).

Common commands:

```bash
python train.py                              # train + watch live
python train.py --no-render                  # faster headless training
python play.py --load checkpoints/final.pt   # watch trained agents
```
