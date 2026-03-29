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
from typing import List, Dict, Tuple, Optional

from env.world import WorldGenerator, World, HIDE_SPOT, DOOR, HEAVY_OBJ, LIGHT_SW
from env.agent import Agent, Team, Action, OBS_DIM, N_ACTIONS

# ── Constants ────────────────────────────────────────────────────────────────
N_HIDERS  = 3
N_SEEKERS = 3
N_AGENTS  = N_HIDERS + N_SEEKERS

PREP_STEPS = 20   # seekers frozen, hiders can hide
MAX_STEPS  = 400

# Agent indices: 0,1,2 = hiders | 3,4,5 = seekers
HIDER_IDS  = list(range(N_HIDERS))
SEEKER_IDS = list(range(N_HIDERS, N_AGENTS))


class HideSeekEnv:
    """
    Multi-agent hide-and-seek environment.

    Interface (gym-like, multi-agent):
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

        self.world:  Optional[World] = None
        self.agents: List[Agent]     = []
        self._step_count   = 0
        self._episode_seed = seed

        # Stats
        self.hiders_caught = 0
        self.seeker_catch_count = {i: 0 for i in SEEKER_IDS}

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
        rooms = self.world.rooms
        mid   = max(1, len(rooms) // 2)
        hider_rooms  = rooms[:mid]
        seeker_rooms = rooms[mid:]

        for i, agent in enumerate(self.agents):
            if agent.team == Team.HIDER:
                room = rng.choice(hider_rooms)
            else:
                room = rng.choice(seeker_rooms)
            inner = room.inner_tiles()
            rng.shuffle(inner)
            r, c = inner[i % len(inner)]
            agent.spawn(r, c)

        self._step_count   = 0
        self.hiders_caught = 0
        self.seeker_catch_count = {i: 0 for i in SEEKER_IDS}

        return self._build_obs(), {}

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action_dict: Dict[int, int]) -> Tuple[Dict, Dict, Dict, Dict]:
        assert self.world is not None, "call reset() first"
        self._step_count += 1
        is_prep = self._step_count <= self.prep_steps

        rewards  = {a.agent_id: 0.0 for a in self.agents}
        step_info: Dict[int, dict] = {}

        # -- Apply individual actions -----------------------------------------
        for agent in self.agents:
            if not agent.alive:
                continue
            # Seekers are frozen during prep phase
            if is_prep and agent.team == Team.SEEKER:
                continue
            act = action_dict.get(agent.agent_id, Action.STAY)
            info = agent.step(act, self.world)
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

        # -- Team-only mechanics ----------------------------------------------
        self._check_heavy_push(rewards)
        self._check_full_blackout(rewards)
        self._check_coordinated_sweep(rewards)

        # -- Catching mechanic ------------------------------------------------
        if not is_prep:
            self._check_catches(rewards)

        # -- Scent decay ------------------------------------------------------
        self.world.step_scent()

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

    def _check_heavy_push(self, rewards: Dict) -> None:
        """
        2+ hiders adjacent to same HEAVY_OBJ + both SIGNAL_A this step.
        Push the object in the direction the majority of signalling agents face.
        """
        signalers = [a for a in self.hiders if a.alive and a.signal_A_this_step]
        if len(signalers) < 2:
            return

        for obj_pos in list(self.world.heavy_positions):
            adjacent = [a for a in signalers
                        if abs(a.row - obj_pos[0]) + abs(a.col - obj_pos[1]) == 1]
            if len(adjacent) < 2:
                continue

            # Push direction = average of agents' facing dirs
            avg_dr = int(round(np.mean([a.last_dir[0] for a in adjacent])))
            avg_dc = int(round(np.mean([a.last_dir[1] for a in adjacent])))
            if avg_dr == 0 and avg_dc == 0:
                avg_dc = 1

            pushed = self.world.push_heavy_obj(obj_pos[0], obj_pos[1], avg_dr, avg_dc)
            if pushed:
                bonus = 15.0
                for a in adjacent:
                    rewards[a.agent_id] += bonus

    def _check_full_blackout(self, rewards: Dict) -> None:
        """
        All 3 alive hiders SIGNAL_B in the same step while each stands on a
        LIGHT_SW tile → toggle all lights off simultaneously.
        """
        alive_hiders = [a for a in self.hiders if a.alive]
        if len(alive_hiders) < 3:
            return
        if not all(a.signal_B_this_step for a in alive_hiders):
            return
        if not all(self.world.grid[a.row, a.col] == LIGHT_SW for a in alive_hiders):
            return

        # Execute blackout
        for room in self.world.rooms:
            room.light_on = False

        bonus = 30.0
        for a in alive_hiders:
            rewards[a.agent_id] += bonus
        # Seekers lose
        for a in self.seekers:
            if a.alive:
                rewards[a.agent_id] -= 15.0

    def _check_coordinated_sweep(self, rewards: Dict) -> None:
        """
        2+ seekers enter the same room from different door tiles in the same step.
        Hiders in that room get stunned briefly.
        """
        alive_seekers = [a for a in self.seekers if a.alive]
        if len(alive_seekers) < 2:
            return

        # Find seekers that just moved into a room this step
        room_entrants: Dict[int, List[Agent]] = {}
        for a in alive_seekers:
            room = self.world.get_room(a.row, a.col)
            if room:
                room_entrants.setdefault(room.room_id, []).append(a)

        for room_id, entrants in room_entrants.items():
            if len(entrants) < 2:
                continue
            # Check they came from different doors (different entry cols/rows)
            positions = [(a.row, a.col) for a in entrants]
            if len(set(positions)) < 2:
                continue

            # Stun hiders in this room
            room = self.world.rooms[room_id]
            stunned = 0
            for h in self.hiders:
                if h.alive and room.contains(h.row, h.col):
                    h.stunned_for = 4
                    stunned += 1

            if stunned > 0:
                sweep_bonus = 60.0
                catch_bonus = sweep_bonus / len(entrants)
                for a in entrants:
                    rewards[a.agent_id] += catch_bonus
                # Hiders penalised
                for h in self.hiders:
                    if h.alive and room.contains(h.row, h.col):
                        rewards[h.agent_id] -= 40.0

    def _check_catches(self, rewards: Dict) -> None:
        """Seeker adjacent to alive hider → hider is caught."""
        for seeker in self.seekers:
            if not seeker.alive:
                continue
            for hider in self.hiders:
                if not hider.alive:
                    continue
                if seeker.is_adjacent_to(hider) and seeker.can_see(hider, self.world):
                    hider.alive  = False
                    hider.caught = True
                    self.hiders_caught += 1
                    self.seeker_catch_count[seeker.agent_id] += 1

                    rewards[hider.agent_id]  -= 100.0
                    rewards[seeker.agent_id] += 150.0

    def _check_seeker_empty_rooms(self, rewards: Dict, step_info: Dict) -> None:
        """Small penalty when seeker enters a room with no hiders."""
        for a in self.seekers:
            if not a.alive:
                continue
            if not step_info.get(a.agent_id, {}).get('moved'):
                continue
            room = self.world.get_room(a.row, a.col)
            if not room:
                continue
            hiders_here = any(h.alive and room.contains(h.row, h.col)
                              for h in self.hiders)
            if not hiders_here:
                rewards[a.agent_id] -= 5.0

    # ── Done & terminal rewards ───────────────────────────────────────────────

    def _build_done(self) -> Dict:
        all_hiders_caught = all(not h.alive for h in self.hiders)
        time_up           = self._step_count >= self.max_steps
        episode_done      = all_hiders_caught or time_up

        done = {a.agent_id: not a.alive or episode_done for a in self.agents}
        done['__all__'] = episode_done
        return done

    def _apply_terminal_rewards(self, rewards: Dict, done_dict: Dict) -> None:
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
        obs = {}
        for agent in self.agents:
            if agent.team == Team.HIDER:
                teammates = [a for a in self.hiders if a.agent_id != agent.agent_id]
            else:
                teammates = [a for a in self.seekers if a.agent_id != agent.agent_id]
            obs[agent.agent_id] = agent.get_observation(self.world, teammates)
        return obs

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
        return {
            'grid':         self.world.grid.copy(),
            'scent_map':    self.world.scent_map.copy(),
            'rooms':        self.world.rooms,
            'agents':       [(a.agent_id, a.row, a.col, a.team.value,
                              a.alive, a.food_count, a.stunned_for > 0)
                             for a in self.agents],
            'step':         self._step_count,
            'max_steps':    self.max_steps,
            'prep':         self._step_count <= self.prep_steps,
            'hiders_caught':self.hiders_caught,
        }

    def __repr__(self) -> str:
        return (f"HideSeekEnv(step={self._step_count}/{self.max_steps}, "
                f"caught={self.hiders_caught}/3)")