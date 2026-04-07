from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class StepInfo:
    score: int
    length: int
    steps: int
    reason: str


class SnakeEnv:
    """Grid-based Snake environment with 3-channel observation.

    Observation: (H, W, 3)
      - channel 0: snake head mask
      - channel 1: snake body mask (excluding head)
      - channel 2: food mask

    Action space (relative):
      0 = straight, 1 = turn right, 2 = turn left
    """

    _DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left

    def __init__(
        self,
        grid_size: int = 20,
        seed: Optional[int] = None,
        max_steps_factor: int = 100,
        loop_visit_threshold: int = 3,
        cell_pixels: int = 24,
    ) -> None:
        self.grid_size = int(grid_size)
        self.max_steps_factor = int(max_steps_factor)
        self.loop_visit_threshold = int(loop_visit_threshold)
        self.cell_pixels = int(cell_pixels)

        self.rng = np.random.default_rng(seed)

        self.snake: deque[tuple[int, int]] = deque()
        self.occupied: set[tuple[int, int]] = set()
        self.food: tuple[int, int] = (0, 0)
        self.direction_idx = 1
        self.score = 0
        self.steps = 0
        self.no_food_steps = 0
        self.visit_counts: dict[tuple[int, int], int] = {}

        self._pygame = None
        self._screen = None
        self._clock = None

    def reset(self) -> np.ndarray:
        c = self.grid_size // 2
        self.snake = deque([(c, c), (c, c - 1), (c, c - 2)])
        self.occupied = set(self.snake)
        self.direction_idx = 1  # right
        self.score = 0
        self.steps = 0
        self.no_food_steps = 0
        self.visit_counts = {(c, c): 1}
        self.food = self._sample_food()
        return self._observation()

    def step(self, action: int):
        if action not in (0, 1, 2):
            raise ValueError(f"Action must be one of {{0,1,2}}, got {action}")

        if action == 1:
            self.direction_idx = (self.direction_idx + 1) % 4
        elif action == 2:
            self.direction_idx = (self.direction_idx - 1) % 4

        dr, dc = self._DIRS[self.direction_idx]
        head_r, head_c = self.snake[0]
        before_dist = abs(head_r - self.food[0]) + abs(head_c - self.food[1])
        new_head = (head_r + dr, head_c + dc)

        self.steps += 1
        self.no_food_steps += 1

        reward = -0.01  # step penalty
        done = False
        truncated = False
        reason = "running"

        # Wall collision
        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            done = True
            reason = "wall_collision"
            reward += -10.0
            return self._observation(), reward, done, truncated, self._info(reason)

        will_grow = new_head == self.food
        tail = self.snake[-1]

        # Self collision (moving into current tail cell is legal if not growing)
        if new_head in self.occupied and (will_grow or new_head != tail):
            done = True
            reason = "self_collision"
            reward += -10.0
            return self._observation(), reward, done, truncated, self._info(reason)

        # Apply move
        self.snake.appendleft(new_head)
        self.occupied.add(new_head)

        if will_grow:
            self.score += 1
            self.no_food_steps = 0
            reward += 10.0
            if len(self.occupied) == self.grid_size * self.grid_size:
                done = True
                reason = "board_filled"
            else:
                self.food = self._sample_food()
            self.visit_counts = {new_head: 1}
        else:
            old_tail = self.snake.pop()
            # If new_head moved into the previous tail cell (legal non-growth move),
            # that cell is still occupied by the new head and must remain in the set.
            if old_tail != new_head:
                self.occupied.remove(old_tail)

            after_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            if after_dist < before_dist:
                reward += 0.1
            elif after_dist > before_dist:
                reward += -0.1

            count = self.visit_counts.get(new_head, 0) + 1
            self.visit_counts[new_head] = count
            if count >= self.loop_visit_threshold:
                reward += -0.5

        if self.steps >= self.max_steps_factor * max(1, len(self.snake)):
            done = True
            truncated = True
            reason = "step_limit"
            reward += -5.0

        return self._observation(), reward, done, truncated, self._info(reason)

    def _sample_food(self) -> tuple[int, int]:
        while True:
            pos = (
                int(self.rng.integers(0, self.grid_size)),
                int(self.rng.integers(0, self.grid_size)),
            )
            if pos not in self.occupied:
                return pos

    def _observation(self) -> np.ndarray:
        obs = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        head_r, head_c = self.snake[0]
        obs[head_r, head_c, 0] = 1.0
        for r, c in list(self.snake)[1:]:
            obs[r, c, 1] = 1.0
        obs[self.food[0], self.food[1], 2] = 1.0
        return obs

    def _info(self, reason: str) -> dict:
        info = StepInfo(
            score=self.score,
            length=len(self.snake),
            steps=self.steps,
            reason=reason,
        )
        return info.__dict__

    def render(self, fps: int = 12) -> None:
        import pygame

        if self._pygame is None:
            self._pygame = pygame
            pygame.init()
            wh = self.grid_size * self.cell_pixels
            self._screen = pygame.display.set_mode((wh, wh))
            pygame.display.set_caption("Snake RL")
            self._clock = pygame.time.Clock()

        assert self._pygame is not None
        assert self._screen is not None
        assert self._clock is not None

        for event in self._pygame.event.get():
            if event.type == self._pygame.QUIT:
                self.close()
                return

        bg = (20, 20, 20)
        grid = (35, 35, 35)
        body = (70, 220, 120)
        head = (20, 180, 80)
        food = (240, 70, 70)

        self._screen.fill(bg)
        w = self.grid_size * self.cell_pixels

        for i in range(self.grid_size + 1):
            x = i * self.cell_pixels
            self._pygame.draw.line(self._screen, grid, (x, 0), (x, w), 1)
            self._pygame.draw.line(self._screen, grid, (0, x), (w, x), 1)

        for idx, (r, c) in enumerate(self.snake):
            color = head if idx == 0 else body
            rect = self._pygame.Rect(
                c * self.cell_pixels + 1,
                r * self.cell_pixels + 1,
                self.cell_pixels - 2,
                self.cell_pixels - 2,
            )
            self._pygame.draw.rect(self._screen, color, rect, border_radius=4)

        fr, fc = self.food
        frect = self._pygame.Rect(
            fc * self.cell_pixels + 2,
            fr * self.cell_pixels + 2,
            self.cell_pixels - 4,
            self.cell_pixels - 4,
        )
        self._pygame.draw.rect(self._screen, food, frect, border_radius=6)

        self._pygame.display.flip()
        self._clock.tick(max(1, int(fps)))

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.quit()
            self._pygame = None
            self._screen = None
            self._clock = None
