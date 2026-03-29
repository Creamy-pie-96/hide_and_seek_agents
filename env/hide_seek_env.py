"""
hide_seek_env.py — Core game loop, reward shaping, and team-only mechanics.

Episode flow:
  1. WorldGenerator.generate() → fresh map
  2. Spawn hiders in one cluster of rooms, seekers in another
  3. Preparation phase (hiders move, seekers frozen) — 20 steps
  4. Main phase — up to max_steps
  5. Done when: all hiders caught OR timer runs out

Reward structure (per agent per step):
  Both teams:
    +0.5   eat real food
    -0.5   eat fake food (seeker only)

  Hiders:
    +1.0   per step alive
    -100   get caught
    +50    survive full episode (shared across alive hiders)
    +20    team executes full blackout
    +10    successfully barricade a door

  Seekers:
    -1.0   per step (urgency pressure)
    +150   catch a hider
    -80    episode ends with hiders alive
    +60    coordinated sweep stuns a hider
    -5     enter empty room (exploration penalty to discourage random walking)

Team-only mechanics:
  HEAVY PUSH   — 2+ hiders adjacent to HEAVY_OBJ, both SIGNAL_A same step
  FULL BLACKOUT— all 3 hiders SIGNAL_B same step + each on LIGHT_SW tile
  COORD SWEEP  — 2+ seekers enter same room from different doors same step
"""

from __future__ import annotations
import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Union, Any

from env.world import WorldGenerator, World, HIDE_SPOT, DOOR, HEAVY_OBJ, LIGHT_SW, EMPTY
from env.agent import Agent, Team, Action, OBS_DIM, N_ACTIONS, GLOBAL_STATE_DIM, MAX_ROOMS_TRACK

# ── Constants ────────────────────────────────────────────────────────────────
N_HIDERS  = 3
N_SEEKERS = 3
N_AGENTS  = N_HIDERS + N_SEEKERS

PREP_STEPS = 20   # seekers frozen, hiders can hide
MAX_STEPS  = 400
ANTI_STAGNATION_START = 25

# Agent indices: 0,1,2 = hiders | 3,4,5 = seekers
HIDER_IDS  = list(range(N_HIDERS))
SEEKER_IDS = list(range(N_HIDERS, N_AGENTS))


class HideSeekEnv:
    """
    Multi-agent hide-and-seek environment.

    Interface (custom parallel multi-agent, gym-like):
        obs_dict, info = env.reset()
        obs_dict, rew_dict, done_dict, info_dict = env.step(action_dict)

    obs_dict   : {agent_id: np.ndarray(OBS_DIM)}
    action_dict: {agent_id: int}
    rew_dict   : {agent_id: float}
    done_dict  : {agent_id: bool, '__all__': bool}
    """

    def __init__(self, width: int = 48, height: int = 48,
                 n_food: int = 14, n_heavy: int = 4,
                 max_steps: int = MAX_STEPS, prep_steps: int = PREP_STEPS,
                 seed: Optional[int] = None):
        self.gen        = WorldGenerator(width, height, n_food, n_heavy)
        self.max_steps  = max_steps
        self.prep_steps = prep_steps
        self.base_seed  = seed

        init_seed = seed if seed is not None else 0
        self.world: World = self.gen.generate(init_seed)
        self.agents: List[Agent]     = []
        self._step_count   = 0
        self._episode_seed = seed

        # Stats
        self.hiders_caught = 0
        self.seeker_catch_count = {i: 0 for i in SEEKER_IDS}

    def _world(self) -> World:
        return self.world

    @staticmethod
    def _sanitize_action(action: Union[int, Action]) -> int:
        try:
            return int(Action(int(action)))
        except (ValueError, TypeError):
            return int(Action.STAY)

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        self._episode_seed = seed if seed is not None else (
            self.base_seed if self.base_seed is not None else random.randint(0, 2**31))
        rng = random.Random(self._episode_seed)

        # Fresh map
        self.world = self.gen.generate(self._episode_seed)

        # Create agents
        self.agents = (
            [Agent(i, Team.HIDER)  for i in HIDER_IDS] +
            [Agent(i, Team.SEEKER) for i in SEEKER_IDS]
        )

        # Spawn: split rooms into two halves — hiders get one, seekers get other
        rooms = self._world().rooms
        if not rooms:
            raise RuntimeError("world generator produced no rooms")

        mid = max(1, len(rooms) // 2)
        hider_rooms = rooms[:mid] if rooms[:mid] else rooms
        seeker_rooms = rooms[mid:] if rooms[mid:] else rooms

        for i, agent in enumerate(self.agents):
            if agent.team == Team.HIDER:
                room = rng.choice(hider_rooms)
            else:
                room = rng.choice(seeker_rooms)
            inner = room.inner_tiles()
            if not inner:
                inner = [room.center]
            rng.shuffle(inner)
            r, c = inner[i % len(inner)]
            agent.spawn(r, c)

        self._step_count   = 0
        self.hiders_caught = 0
        self.seeker_catch_count = {i: 0 for i in SEEKER_IDS}

        return self._build_obs(), {}

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action_dict: Dict[int, int]) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[Union[int, str], bool], Dict[int, dict]]:
        world = self._world()
        self._step_count += 1
        is_prep = self._step_count <= self.prep_steps

        rewards  = {a.agent_id: 0.0 for a in self.agents}
        step_info: Dict[int, dict] = {}
        prev_pos: Dict[int, Tuple[int, int]] = {
            a.agent_id: (a.row, a.col) for a in self.agents
        }

        # -- Apply individual actions -----------------------------------------
        for agent in self.agents:
            if not agent.alive:
                continue
            # Seekers are frozen during prep phase
            if is_prep and agent.team == Team.SEEKER:
                continue
            raw_act = action_dict.get(agent.agent_id, int(Action.STAY))
            act = self._sanitize_action(raw_act)
            info = agent.step(act, world)
            info['prev_pos'] = prev_pos[agent.agent_id]
            step_info[agent.agent_id] = info

        # -- Per-step base rewards --------------------------------------------
        for agent in self.agents:
            if not agent.alive:
                continue
            if agent.team == Team.HIDER:
                rewards[agent.agent_id] += 1.0        # survival reward
            else:
                rewards[agent.agent_id] -= 1.0        # urgency

            # Food rewards
            info = step_info.get(agent.agent_id, {})
            if info.get('ate_food'):
                rewards[agent.agent_id] += 0.5
            if info.get('ate_fake') and agent.team == Team.SEEKER:
                rewards[agent.agent_id] -= 0.5
            if info.get('barricaded') and agent.team == Team.HIDER:
                rewards[agent.agent_id] += 10.0
            if (not is_prep and agent.team == Team.HIDER and
                    agent.stagnation_steps > ANTI_STAGNATION_START):
                rewards[agent.agent_id] -= min(1.0, 0.02 * (agent.stagnation_steps - ANTI_STAGNATION_START))

        # -- Team-only mechanics ----------------------------------------------
        self._check_heavy_push(rewards)
        self._check_full_blackout(rewards)
        self._check_coordinated_sweep(rewards, step_info)

        # -- Catching mechanic ------------------------------------------------
        if not is_prep:
            self._check_catches(rewards)

        # -- Scent decay ------------------------------------------------------
        world.step_scent()

        # -- Exploration penalty for seekers (empty room entry) ---------------
        self._check_seeker_empty_rooms(rewards, step_info)

        # -- Done conditions --------------------------------------------------
        done_dict = self._build_done()

        if done_dict['__all__']:
            self._apply_terminal_rewards(rewards, done_dict)

        obs_dict  = self._build_obs()
        info_dict = {a.agent_id: step_info.get(a.agent_id, {}) for a in self.agents}

        return obs_dict, rewards, done_dict, info_dict

    # ── Team mechanics ────────────────────────────────────────────────────────

    def _check_heavy_push(self, rewards: Dict[int, float]) -> None:
        """
        2+ hiders adjacent to same HEAVY_OBJ + both SIGNAL_A this step.
        Push the object in the direction the majority of signalling agents face.
        """
        signalers = [a for a in self.hiders if a.alive and a.signal_A_this_step]
        if len(signalers) < 2:
            return

        world = self._world()
        for obj_pos in list(world.heavy_positions):
            adjacent = [a for a in signalers
                        if abs(a.row - obj_pos[0]) + abs(a.col - obj_pos[1]) == 1]
            if len(adjacent) < 2:
                continue

            # Push direction = average of agents' facing dirs
            avg_dr = int(round(np.mean([a.last_dir[0] for a in adjacent])))
            avg_dc = int(round(np.mean([a.last_dir[1] for a in adjacent])))
            if avg_dr == 0 and avg_dc == 0:
                avg_dc = 1

            pushed = world.push_heavy_obj(obj_pos[0], obj_pos[1], avg_dr, avg_dc)
            if pushed:
                bonus = 15.0
                for a in adjacent:
                    rewards[a.agent_id] += bonus

    def _check_full_blackout(self, rewards: Dict[int, float]) -> None:
        """
        All 3 alive hiders SIGNAL_B in the same step while each stands on a
        LIGHT_SW tile → toggle all lights off simultaneously.
        """
        world = self._world()
        alive_hiders = [a for a in self.hiders if a.alive]
        if len(alive_hiders) < 3:
            return
        if not all(a.signal_B_this_step for a in alive_hiders):
            return
        if not all(world.grid[a.row, a.col] == LIGHT_SW for a in alive_hiders):
            return

        room_ids: List[int] = []
        for a in alive_hiders:
            room = world.get_room(a.row, a.col)
            if room is not None:
                room_ids.append(room.room_id)
        if len(room_ids) != 3 or len(set(room_ids)) != 3:
            return

        # Execute blackout
        for room in world.rooms:
            room.light_on = False

        contributors = [a for a in alive_hiders if world.grid[a.row, a.col] == LIGHT_SW and a.signal_B_this_step]
        if not contributors:
            contributors = alive_hiders

        team_bonus = 6.0
        contributor_bonus = 24.0 / len(contributors)
        for a in alive_hiders:
            rewards[a.agent_id] += team_bonus
        for a in contributors:
            rewards[a.agent_id] += contributor_bonus
        # Seekers lose
        for a in self.seekers:
            if a.alive:
                rewards[a.agent_id] -= 15.0

    def _check_coordinated_sweep(self, rewards: Dict[int, float], step_info: Dict[int, dict]) -> None:
        """
        2+ seekers enter the same room from different door tiles in the same step.
        Hiders in that room get stunned briefly.
        """
        world = self._world()
        alive_seekers = [a for a in self.seekers if a.alive]
        if len(alive_seekers) < 2:
            return

        door_positions = set(world.door_positions)
        room_entrants: Dict[int, List[Tuple[Agent, Tuple[int, int]]]] = {}
        for a in alive_seekers:
            info = step_info.get(a.agent_id, {})
            if not info.get('moved', False):
                continue
            prev_r, prev_c = info.get('prev_pos', (a.row, a.col))
            room = world.get_room(a.row, a.col)
            if room is None:
                continue

            entry_door: Optional[Tuple[int, int]] = None
            if (prev_r, prev_c) in door_positions:
                entry_door = (prev_r, prev_c)
            elif (a.row, a.col) in door_positions:
                entry_door = (a.row, a.col)

            if entry_door is not None:
                room_entrants.setdefault(room.room_id, []).append((a, entry_door))

        for room_id, entrants in room_entrants.items():
            if len(entrants) < 2:
                continue
            distinct_doors = {door for _, door in entrants}
            if len(distinct_doors) < 2:
                continue

            # Stun hiders in this room
            room = world.rooms[room_id]
            stunned = 0
            for h in self.hiders:
                if h.alive and room.contains(h.row, h.col):
                    h.stunned_for = 4
                    stunned += 1

            if stunned > 0:
                sweep_bonus = 60.0
                catch_bonus = sweep_bonus / len(entrants)
                for a, _ in entrants:
                    rewards[a.agent_id] += catch_bonus
                # Hiders penalised
                for h in self.hiders:
                    if h.alive and room.contains(h.row, h.col):
                        rewards[h.agent_id] -= 40.0

    def _check_catches(self, rewards: Dict[int, float]) -> None:
        """Seeker adjacent to alive hider → hider is caught."""
        world = self._world()
        ping = world.objects.last_light_ping
        for seeker in self.seekers:
            if not seeker.alive:
                continue
            for hider in self.hiders:
                if not hider.alive:
                    continue
                ping_reveal = False
                if ping is not None:
                    pr, pc = ping.pos
                    seeker_near_ping = abs(seeker.row - pr) + abs(seeker.col - pc) <= 8
                    hider_in_ping_room = world.get_room(hider.row, hider.col) == world.get_room(pr, pc)
                    ping_reveal = seeker_near_ping and hider_in_ping_room

                relay_visible = any(
                    ally.alive
                    and ally.agent_id != seeker.agent_id
                    and seeker.is_adjacent_to(ally)
                    and ally.can_see(hider, world)
                    for ally in self.seekers
                )

                if seeker.is_adjacent_to(hider) and (seeker.can_see(hider, world) or relay_visible or ping_reveal):
                    hider.alive  = False
                    hider.caught = True
                    self.hiders_caught += 1
                    self.seeker_catch_count[seeker.agent_id] += 1

                    rewards[hider.agent_id]  -= 100.0
                    rewards[seeker.agent_id] += 150.0

    def _check_seeker_empty_rooms(self, rewards: Dict[int, float], step_info: Dict[int, dict]) -> None:
        """Small penalty when seeker enters a room with no hiders."""
        world = self._world()
        for a in self.seekers:
            if not a.alive:
                continue
            if not step_info.get(a.agent_id, {}).get('moved'):
                continue
            room = world.get_room(a.row, a.col)
            if not room:
                continue
            hiders_here = any(h.alive and room.contains(h.row, h.col)
                              for h in self.hiders)
            if not hiders_here:
                rewards[a.agent_id] -= 5.0

    # ── Done & terminal rewards ───────────────────────────────────────────────

    def _build_done(self) -> Dict[Union[int, str], bool]:
        all_hiders_caught = all(not h.alive for h in self.hiders)
        time_up           = self._step_count >= self.max_steps
        episode_done      = all_hiders_caught or time_up

        done: Dict[Union[int, str], bool] = {
            a.agent_id: (not a.alive or episode_done) for a in self.agents
        }
        done['__all__'] = episode_done
        return done

    def _apply_terminal_rewards(self, rewards: Dict[int, float], done_dict: Dict[Union[int, str], bool]) -> None:
        all_caught = all(not h.alive for h in self.hiders)
        if not all_caught:
            # Hiders won — time ran out
            for h in self.hiders:
                if h.alive:
                    rewards[h.agent_id] += 50.0
            for s in self.seekers:
                if s.alive:
                    rewards[s.agent_id] -= 80.0
        else:
            # Seekers won — all hiders caught
            for s in self.seekers:
                if s.alive:
                    rewards[s.agent_id] += 30.0   # bonus for clean sweep

    # ── Observation builder ───────────────────────────────────────────────────

    def _build_obs(self) -> Dict[int, np.ndarray]:
        world = self._world()
        obs = {}
        for agent in self.agents:
            if agent.team == Team.HIDER:
                teammates = [a for a in self.hiders if a.agent_id != agent.agent_id]
            else:
                teammates = [a for a in self.seekers if a.agent_id != agent.agent_id]
            obs[agent.agent_id] = agent.get_observation(world, teammates)
        return obs

    def get_action_mask(self, agent_id: int) -> np.ndarray:
        """Binary valid-action mask for action-level operational constraints."""
        mask = np.zeros(N_ACTIONS, dtype=np.float32)
        world = self._world()
        agent = self.agents[agent_id]

        if (not agent.alive) or agent.stunned_for > 0:
            mask[int(Action.STAY)] = 1.0
            return mask

        mask[int(Action.STAY)] = 1.0
        # movement
        for act, (dr, dc) in {
            Action.UP: (-1, 0),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            Action.RIGHT: (0, 1),
        }.items():
            nr, nc = agent.row + dr, agent.col + dc
            if world.is_walkable(nr, nc):
                mask[int(act)] = 1.0

        if agent.light_cooldown == 0 and world.grid[agent.row, agent.col] == LIGHT_SW:
            mask[int(Action.TOGGLE_LIGHT)] = 1.0

        if agent.barricade_cooldown == 0:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                r, c = agent.row + dr, agent.col + dc
                if 0 <= r < world.height and 0 <= c < world.width and world.grid[r, c] == DOOR:
                    mask[int(Action.BARRICADE)] = 1.0
                    break

        if agent.team == Team.HIDER and agent.fake_food_cooldown == 0 and world.grid[agent.row, agent.col] == EMPTY:
            mask[int(Action.DROP_FAKE_FOOD)] = 1.0

        if agent.scent_cooldown == 0:
            mask[int(Action.DROP_SCENT)] = 1.0

        # team signals always valid for alive agents
        mask[int(Action.SIGNAL_A)] = 1.0
        mask[int(Action.SIGNAL_B)] = 1.0
        return mask

    def get_action_masks(self) -> Dict[int, np.ndarray]:
        return {a.agent_id: self.get_action_mask(a.agent_id) for a in self.agents}

    def build_global_state(self) -> np.ndarray:
        """Fixed-size global state for centralized critic."""
        world = self._world()
        vec = np.zeros(GLOBAL_STATE_DIM, dtype=np.float32)
        ptr = 0

        # per-agent features: row, col, alive, team, stunned, food
        for a in self.agents:
            vec[ptr] = a.row / max(1.0, world.height - 1)
            vec[ptr + 1] = a.col / max(1.0, world.width - 1)
            vec[ptr + 2] = float(a.alive)
            vec[ptr + 3] = float(a.team.value)
            vec[ptr + 4] = float(a.stunned_for > 0)
            vec[ptr + 5] = min(1.0, a.food_count / 10.0)
            ptr += 6

        # room light states (padded)
        for i in range(min(MAX_ROOMS_TRACK, len(world.rooms))):
            vec[ptr + i] = 1.0 if world.rooms[i].light_on else 0.0
        ptr += MAX_ROOMS_TRACK

        # misc summary
        ping = world.objects.last_light_ping
        vec[ptr] = self._step_count / max(1.0, self.max_steps)
        vec[ptr + 1] = self.hiders_caught / 3.0
        vec[ptr + 2] = sum(1 for h in self.hiders if h.alive) / 3.0
        vec[ptr + 3] = sum(1 for s in self.seekers if s.alive) / 3.0
        vec[ptr + 4] = 0.0 if ping is None else ping.pos[0] / max(1.0, world.height - 1)
        vec[ptr + 5] = 0.0 if ping is None else ping.pos[1] / max(1.0, world.width - 1)
        return vec

    # ── Shortcuts ─────────────────────────────────────────────────────────────

    @property
    def hiders(self) -> List[Agent]:
        return [a for a in self.agents if a.team == Team.HIDER]

    @property
    def seekers(self) -> List[Agent]:
        return [a for a in self.agents if a.team == Team.SEEKER]

    # ── Spaces ────────────────────────────────────────────────────────────────

    @property
    def observation_dim(self) -> int:
        return OBS_DIM

    @property
    def action_dim(self) -> int:
        return N_ACTIONS

    # ── Render helper ─────────────────────────────────────────────────────────

    def get_render_state(self) -> dict:
        """Return everything the renderer needs each frame."""
        world = self._world()
        return {
            'grid':         world.grid.copy(),
            'scent_map':    world.scent_map.copy(),
            'rooms':        world.rooms,
            'tile_to_room': dict(world.tile_to_room),
            'light_ping':   None if world.objects.last_light_ping is None else {
                'pos': world.objects.last_light_ping.pos,
                'ttl': world.objects.last_light_ping.ttl,
            },
            'agents':       [(a.agent_id, a.row, a.col, a.team.value,
                              a.alive, a.food_count, a.stunned_for > 0)
                             for a in self.agents],
            'step':         self._step_count,
            'max_steps':    self.max_steps,
            'prep':         self._step_count <= self.prep_steps,
            'hiders_caught':self.hiders_caught,
        }

    def get_serializable_render_state(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable render state.

        This is intentionally lightweight and renderer-agnostic so it can be
        used for replay storage and offline visualization.
        """
        world = self._world()
        room_lights = {int(room.room_id): bool(room.light_on) for room in world.rooms}
        tile_to_room = [[int(r), int(c), int(room_id)]
                        for (r, c), room_id in world.tile_to_room.items()]

        return {
            'grid': world.grid.astype(int).tolist(),
            'scent_map': world.scent_map.astype(float).tolist(),
            'room_lights': room_lights,
            'tile_to_room': tile_to_room,
            'light_ping': None if world.objects.last_light_ping is None else {
                'pos': [int(world.objects.last_light_ping.pos[0]), int(world.objects.last_light_ping.pos[1])],
                'ttl': int(world.objects.last_light_ping.ttl),
            },
            'agents': [
                {
                    'id': int(a.agent_id),
                    'row': int(a.row),
                    'col': int(a.col),
                    'team': int(a.team.value),
                    'alive': bool(a.alive),
                    'food': int(a.food_count),
                    'stunned': bool(a.stunned_for > 0),
                }
                for a in self.agents
            ],
            'step': int(self._step_count),
            'max_steps': int(self.max_steps),
            'prep': bool(self._step_count <= self.prep_steps),
            'hiders_caught': int(self.hiders_caught),
            'width': int(world.width),
            'height': int(world.height),
        }

    def __repr__(self) -> str:
        return (f"HideSeekEnv(step={self._step_count}/{self.max_steps}, "
                f"caught={self.hiders_caught}/3)")