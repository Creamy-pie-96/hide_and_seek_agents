"""
memory.py — Rollout memory for MAPPO training.

Provides:
- per-agent transition storage
- alive masking to avoid dead-agent pollution
- tensor conversion helpers for PPO updates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


@dataclass
class Transition:
	obs: np.ndarray
	action: int
	log_prob: float
	value: float
	reward: float
	done: bool
	alive: bool
	value_obs: np.ndarray
	action_mask: np.ndarray
	pad: bool


@dataclass
class AgentRollout:
	transitions: List[Transition] = field(default_factory=list)

	def add(self, t: Transition) -> None:
		self.transitions.append(t)

	def clear(self) -> None:
		self.transitions.clear()

	def __len__(self) -> int:
		return len(self.transitions)

	def arrays(self) -> Dict[str, np.ndarray]:
		if not self.transitions:
			return {
				'obs': np.empty((0, 0), dtype=np.float32),
				'value_obs': np.empty((0, 0), dtype=np.float32),
				'actions': np.empty((0,), dtype=np.int64),
				'log_probs': np.empty((0,), dtype=np.float32),
				'values': np.empty((0,), dtype=np.float32),
				'rewards': np.empty((0,), dtype=np.float32),
				'dones': np.empty((0,), dtype=np.bool_),
				'alive': np.empty((0,), dtype=np.bool_),
				'action_mask': np.empty((0, 0), dtype=np.float32),
				'pad': np.empty((0,), dtype=np.bool_),
			}

		obs = np.stack([t.obs for t in self.transitions]).astype(np.float32)
		value_obs = np.stack([t.value_obs for t in self.transitions]).astype(np.float32)
		actions = np.array([t.action for t in self.transitions], dtype=np.int64)
		log_probs = np.array([t.log_prob for t in self.transitions], dtype=np.float32)
		values = np.array([t.value for t in self.transitions], dtype=np.float32)
		rewards = np.array([t.reward for t in self.transitions], dtype=np.float32)
		dones = np.array([t.done for t in self.transitions], dtype=np.bool_)
		alive = np.array([t.alive for t in self.transitions], dtype=np.bool_)
		action_mask = np.stack([t.action_mask for t in self.transitions]).astype(np.float32)
		pad = np.array([t.pad for t in self.transitions], dtype=np.bool_)

		return {
			'obs': obs,
			'value_obs': value_obs,
			'actions': actions,
			'log_probs': log_probs,
			'values': values,
			'rewards': rewards,
			'dones': dones,
			'alive': alive,
			'action_mask': action_mask,
			'pad': pad,
		}


class TeamRolloutMemory:
	"""Rollout memory keyed by agent ID."""

	def __init__(self, agent_ids: Sequence[int]):
		self._agent_ids = list(agent_ids)
		self._store: Dict[int, AgentRollout] = {aid: AgentRollout() for aid in self._agent_ids}

	@property
	def agent_ids(self) -> List[int]:
		return list(self._agent_ids)

	def clear(self) -> None:
		for b in self._store.values():
			b.clear()

	def add(
		self,
		agent_id: int,
		obs: np.ndarray,
		action: int,
		log_prob: float,
		value: float,
		reward: float,
		done: bool,
		alive: bool,
		value_obs: np.ndarray,
		action_mask: np.ndarray,
		pad: bool,
	) -> None:
		self._store[agent_id].add(
			Transition(
				obs=obs,
				action=int(action),
				log_prob=float(log_prob),
				value=float(value),
				reward=float(reward),
				done=bool(done),
				alive=bool(alive),
				value_obs=value_obs,
				action_mask=action_mask,
				pad=bool(pad),
			)
		)

	def arrays_for(self, agent_id: int) -> Dict[str, np.ndarray]:
		return self._store[agent_id].arrays()

	def concat_team(self, team_ids: Sequence[int]) -> Dict[str, np.ndarray]:
		blocks: Dict[str, List[np.ndarray]] = {
			'obs': [],
			'value_obs': [],
			'actions': [],
			'log_probs': [],
			'values': [],
			'rewards': [],
			'dones': [],
			'alive': [],
			'action_mask': [],
			'pad': [],
		}
		for aid in team_ids:
			arr = self.arrays_for(aid)
			if arr['obs'].shape[0] == 0:
				continue
			for k in blocks:
				blocks[k].append(arr[k])

		out: Dict[str, np.ndarray] = {}
		for k, vals in blocks.items():
			if not vals:
				if k in ('obs', 'value_obs'):
					out[k] = np.empty((0, 0), dtype=np.float32)
				elif k in ('actions',):
					out[k] = np.empty((0,), dtype=np.int64)
				elif k in ('dones', 'alive', 'pad'):
					out[k] = np.empty((0,), dtype=np.bool_)
				elif k in ('action_mask',):
					out[k] = np.empty((0, 0), dtype=np.float32)
				else:
					out[k] = np.empty((0,), dtype=np.float32)
			else:
				out[k] = np.concatenate(vals, axis=0)
		return out


def to_torch(arr: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
	return torch.tensor(arr, dtype=dtype, device=device)

