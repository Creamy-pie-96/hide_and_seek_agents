"""
objects.py — Typed object models and mutable object state for the world.

This module provides a clean object layer above raw tile IDs so world/env logic
can evolve without scattering ad-hoc mutable state across files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple
import numpy as np


Coord = Tuple[int, int]


@dataclass(frozen=True)
class Food:
	pos: Coord


@dataclass(frozen=True)
class FakeFood:
	pos: Coord


@dataclass(frozen=True)
class LightSwitch:
	pos: Coord
	room_id: int


@dataclass(frozen=True)
class HeavyObject:
	pos: Coord


@dataclass
class RevealPing:
	pos: Coord
	ttl: int


@dataclass
class ObjectState:
	"""
	Mutable per-episode object state.

	Notes:
	- `barricaded_doors` tracks blocked door coordinates.
	- `fake_food` tracks currently active fake-food coordinates.
	- scent is represented as continuous map + TTL map for decay.
	- `last_light_ping` stores temporary seeker-visible location ping.
	"""

	width: int
	height: int
	barricaded_doors: Set[Coord] = field(default_factory=set)
	fake_food: Set[Coord] = field(default_factory=set)
	scent_map: np.ndarray = field(init=False)
	scent_ttl: np.ndarray = field(init=False)
	last_light_ping: Optional[RevealPing] = None

	def __post_init__(self) -> None:
		self.scent_map = np.zeros((self.height, self.width), dtype=np.float32)
		self.scent_ttl = np.zeros((self.height, self.width), dtype=np.int32)

	def reset(self) -> None:
		self.barricaded_doors.clear()
		self.fake_food.clear()
		self.scent_map[:] = 0.0
		self.scent_ttl[:] = 0
		self.last_light_ping = None

	def set_light_ping(self, pos: Coord, ttl: int = 8) -> None:
		self.last_light_ping = RevealPing(pos=pos, ttl=ttl)

	def step_light_ping(self) -> None:
		if self.last_light_ping is None:
			return
		self.last_light_ping.ttl -= 1
		if self.last_light_ping.ttl <= 0:
			self.last_light_ping = None

	def add_fake_food(self, pos: Coord) -> None:
		self.fake_food.add(pos)

	def remove_fake_food(self, pos: Coord) -> None:
		self.fake_food.discard(pos)

	def add_barricade(self, pos: Coord) -> None:
		self.barricaded_doors.add(pos)

	def remove_barricade(self, pos: Coord) -> None:
		self.barricaded_doors.discard(pos)

	def drop_scent(self, r: int, c: int, strength: float = 1.0, ttl: int = 15) -> None:
		self.scent_map[r, c] = float(strength)
		self.scent_ttl[r, c] = int(ttl)

	def decay_scent(self, decay: float = 0.85) -> None:
		mask = self.scent_ttl > 0
		self.scent_ttl[mask] -= 1
		self.scent_map[mask] *= float(decay)
		self.scent_map[self.scent_ttl == 0] = 0.0

	def scent_window(self, row: int, col: int, radius: int) -> np.ndarray:
		"""
		Returns a local scent patch in [0, 1] with zero padding outside bounds.
		Shape: (2*radius+1, 2*radius+1)
		"""
		size = 2 * radius + 1
		out = np.zeros((size, size), dtype=np.float32)
		for dr in range(-radius, radius + 1):
			for dc in range(-radius, radius + 1):
				r, c = row + dr, col + dc
				if 0 <= r < self.height and 0 <= c < self.width:
					out[dr + radius, dc + radius] = self.scent_map[r, c]
		return out

