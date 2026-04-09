from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class StepInfo:
    score: int
    opponent_score: int
    length: int
    steps: int
    reason: str


class SnakeEnv:
    """Grid-based Snake environment with optional static opponent.

    Observation: (H, W, 4)
      - channel 0: snake head mask
      - channel 1: snake body mask (excluding head)
      - channel 2: food mask
      - channel 3: opponent head mask

    Action space (relative):
      0 = straight, 1 = turn right, 2 = turn left
    """

    _DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def __init__(
        self,
        grid_size: int = 20,
        seed: Optional[int] = None,
        max_steps_factor: int = 100,
        loop_visit_threshold: int = 3,
        loop_visit_penalty: float = -0.03,
        food_reward: float = 15.0,
        death_penalty: float = -8.0,
        distance_reward_toward: float = 0.05,
        distance_penalty_away: float = -0.1,
        stagnation_penalty: float = 0.0,
        survival_reward: float = 0.002,
        idle_step_coeff: float = 0.0,
        hunger_exp_base: float = 0.0,
        hunger_exp_gamma: float = 1.01,
        hunger_exp_max: float = 0.5,
        step_limit_penalty: float = -1.0,
        starvation_steps_factor: int = 60,
        starvation_penalty: float = -6.0,
        wall_follow_threshold: int = 10,
        wall_follow_penalty: float = -0.04,
        obstacle_count: int = 0,
        moving_obstacles: bool = False,
        obstacle_move_period: int = 8,
        moving_food: bool = False,
        food_move_prob: float = 0.15,
        food_spawn_radius: int | None = None,
        opponent_mode: str = "none",
        opponent_food_penalty: float = -2.0,
        opponent_random_prob: float = 0.15,
        terminal_win_reward: float = 8.0,
        terminal_loss_penalty: float = -8.0,
        cell_pixels: int = 24,
    ) -> None:
        self.grid_size = int(grid_size)
        self.max_steps_factor = int(max_steps_factor)
        self.loop_visit_threshold = int(loop_visit_threshold)
        self.loop_visit_penalty = float(loop_visit_penalty)
        self.food_reward = float(food_reward)
        self.death_penalty = float(death_penalty)
        self.distance_reward_toward = float(distance_reward_toward)
        self.distance_penalty_away = float(distance_penalty_away)
        self.stagnation_penalty = float(stagnation_penalty)
        self.survival_reward = float(survival_reward)
        self.idle_step_coeff = float(idle_step_coeff)
        self.hunger_exp_base = float(hunger_exp_base)
        self.hunger_exp_gamma = float(hunger_exp_gamma)
        self.hunger_exp_max = float(hunger_exp_max)
        self.step_limit_penalty = float(step_limit_penalty)
        self.starvation_steps_factor = max(1, int(starvation_steps_factor))
        self.starvation_penalty = float(starvation_penalty)
        self.wall_follow_threshold = max(1, int(wall_follow_threshold))
        self.wall_follow_penalty = float(wall_follow_penalty)
        self.obstacle_count = max(0, int(obstacle_count))
        self.moving_obstacles = bool(moving_obstacles)
        self.obstacle_move_period = max(1, int(obstacle_move_period))
        self.moving_food = bool(moving_food)
        self.food_move_prob = float(np.clip(food_move_prob, 0.0, 1.0))
        self.food_spawn_radius = None if food_spawn_radius is None else int(food_spawn_radius)
        self.opponent_mode = str(opponent_mode)
        self.opponent_food_penalty = float(opponent_food_penalty)
        self.opponent_random_prob = float(np.clip(opponent_random_prob, 0.0, 1.0))
        self.terminal_win_reward = float(terminal_win_reward)
        self.terminal_loss_penalty = float(terminal_loss_penalty)
        self.cell_pixels = int(cell_pixels)

        self.rng = np.random.default_rng(seed)

        self.snake: deque[tuple[int, int]] = deque()
        self.occupied: set[tuple[int, int]] = set()
        self.food: tuple[int, int] = (0, 0)
        self.direction_idx = 1
        self.score = 0
        self.opponent_score = 0
        self.steps = 0
        self.no_food_steps = 0
        self.visit_counts: dict[tuple[int, int], int] = {}
        self.opponent_pos: tuple[int, int] = (0, 0)
        self.opponent_dir_idx = 3
        self.wall_follow_steps = 0
        self.blocks: set[tuple[int, int]] = set()

        self._pygame = None
        self._screen = None
        self._clock = None

    @property
    def obs_channels(self) -> int:
        return 5

    @property
    def has_opponent(self) -> bool:
        return self.opponent_mode != "none"

    def reset(self) -> np.ndarray:
        c = self.grid_size // 2
        self.snake = deque([(c, c), (c, c - 1), (c, c - 2)])
        self.occupied = set(self.snake)
        self.direction_idx = 1
        self.score = 0
        self.opponent_score = 0
        self.steps = 0
        self.no_food_steps = 0
        self.visit_counts = {(c, c): 1}
        self.opponent_dir_idx = 3
        self.opponent_pos = self._sample_opponent_start()
        self.wall_follow_steps = 0
        self.blocks = self._sample_blocks(self.obstacle_count)
        self.food = self._sample_food()
        return self._observation()

    def _sample_blocks(self, count: int) -> set[tuple[int, int]]:
        blocks: set[tuple[int, int]] = set()
        target = max(0, int(count))
        attempts = 0
        max_attempts = max(200, target * 40)
        while len(blocks) < target and attempts < max_attempts:
            attempts += 1
            pos = (
                int(self.rng.integers(0, self.grid_size)),
                int(self.rng.integers(0, self.grid_size)),
            )
            if pos in self.occupied:
                continue
            if self.has_opponent and pos == self.opponent_pos:
                continue
            if pos == self.food:
                continue
            blocks.add(pos)
        return blocks

    def _blocked_cells(self) -> set[tuple[int, int]]:
        return set(self.blocks)

    def _try_move_food(self) -> None:
        if not self.moving_food:
            return
        if float(self.rng.random()) >= self.food_move_prob:
            return
        candidates: list[tuple[int, int]] = []
        for dr, dc in self._DIRS:
            nr, nc = self.food[0] + dr, self.food[1] + dc
            if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
                continue
            nxt = (nr, nc)
            if nxt in self.occupied or nxt in self.blocks:
                continue
            if self.has_opponent and nxt == self.opponent_pos:
                continue
            candidates.append(nxt)
        if candidates:
            idx = int(self.rng.integers(0, len(candidates)))
            self.food = candidates[idx]

    def _move_blocks(self) -> None:
        if not self.moving_obstacles:
            return
        if self.steps <= 0 or (self.steps % self.obstacle_move_period) != 0:
            return
        moved: set[tuple[int, int]] = set()
        for br, bc in list(self.blocks):
            options: list[tuple[int, int]] = [(br, bc)]
            for dr, dc in self._DIRS:
                nr, nc = br + dr, bc + dc
                if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
                    continue
                nxt = (nr, nc)
                if nxt in self.occupied:
                    continue
                if self.has_opponent and nxt == self.opponent_pos:
                    continue
                if nxt == self.food:
                    continue
                if nxt in moved:
                    continue
                options.append(nxt)
            idx = int(self.rng.integers(0, len(options)))
            moved.add(options[idx])
        self.blocks = moved

    def _relative_to_abs_dir(self, base_dir: int, rel_action: int) -> int:
        if rel_action == 1:
            return (base_dir + 1) % 4
        if rel_action == 2:
            return (base_dir - 1) % 4
        return base_dir

    def _heuristic_opponent_action(self) -> int:
        valid: list[int] = []
        best = 0
        best_dist = 10**9
        for rel in (0, 1, 2):
            d_idx = self._relative_to_abs_dir(self.opponent_dir_idx, rel)
            dr, dc = self._DIRS[d_idx]
            nr, nc = self.opponent_pos[0] + dr, self.opponent_pos[1] + dc
            if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
                continue
            if (nr, nc) in self.occupied:
                continue
            if (nr, nc) in self.blocks:
                continue
            valid.append(rel)
            dist = abs(nr - self.food[0]) + abs(nc - self.food[1])
            if dist < best_dist:
                best_dist = dist
                best = rel
        if not valid:
            return 0
        if float(self.rng.random()) < self.opponent_random_prob:
            idx = int(self.rng.integers(0, len(valid)))
            return int(valid[idx])
        return best

    def _sample_opponent_start(self) -> tuple[int, int]:
        while True:
            pos = (
                int(self.rng.integers(0, self.grid_size)),
                int(self.rng.integers(0, self.grid_size)),
            )
            if pos not in self.occupied:
                return pos

    def _apply_terminal_outcome_reward(self, reward: float, reason: str) -> float:
        if not self.has_opponent:
            return float(reward)
        if reason not in ("step_limit", "board_filled"):
            return float(reward)
        score_diff = int(self.score - self.opponent_score)
        if score_diff >= 2:
            return float(reward + self.terminal_win_reward)
        if score_diff <= -2:
            return float(reward + self.terminal_loss_penalty)
        return float(reward)

    def step(self, action: int, opponent_action: int | None = None):
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

        if self.has_opponent:
            if opponent_action is None or self.opponent_mode == "heuristic":
                opponent_action = self._heuristic_opponent_action()
            if opponent_action not in (0, 1, 2):
                opponent_action = 0
            opp_abs = self._relative_to_abs_dir(self.opponent_dir_idx, int(opponent_action))
            odr, odc = self._DIRS[opp_abs]
            new_opp = (self.opponent_pos[0] + odr, self.opponent_pos[1] + odc)
            if (
                not (0 <= new_opp[0] < self.grid_size and 0 <= new_opp[1] < self.grid_size)
                or (new_opp in self.occupied)
                or (new_opp in self.blocks)
            ):
                new_opp = self.opponent_pos
            else:
                self.opponent_dir_idx = opp_abs
        else:
            new_opp = self.opponent_pos

        self.steps += 1
        self.no_food_steps += 1

        reward = self.survival_reward
        done = False
        truncated = False
        reason = "running"

        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            done = True
            reason = "wall_collision"
            reward += self.death_penalty
            reward = self._apply_terminal_outcome_reward(reward, reason)
            return self._observation(), reward, done, truncated, self._info(reason)

        if new_head in self.blocks:
            done = True
            reason = "block_collision"
            reward += self.death_penalty
            reward = self._apply_terminal_outcome_reward(reward, reason)
            return self._observation(), reward, done, truncated, self._info(reason)

        if self.has_opponent and new_head == new_opp:
            done = True
            reason = "opponent_collision"
            reward += self.death_penalty
            reward = self._apply_terminal_outcome_reward(reward, reason)
            return self._observation(), reward, done, truncated, self._info(reason)

        player_gets_food = new_head == self.food
        opponent_gets_food = self.has_opponent and (new_opp == self.food)
        will_grow = player_gets_food and not opponent_gets_food
        tail = self.snake[-1]

        if new_head in self.occupied and (will_grow or new_head != tail):
            done = True
            reason = "self_collision"
            reward += self.death_penalty
            reward = self._apply_terminal_outcome_reward(reward, reason)
            return self._observation(), reward, done, truncated, self._info(reason)

        self.snake.appendleft(new_head)
        self.occupied.add(new_head)
        self.opponent_pos = new_opp

        on_border = (
            new_head[0] == 0
            or new_head[0] == self.grid_size - 1
            or new_head[1] == 0
            or new_head[1] == self.grid_size - 1
        )
        self.wall_follow_steps = (self.wall_follow_steps + 1) if on_border else 0

        if player_gets_food and not opponent_gets_food:
            self.score += 1
            self.no_food_steps = 0
            reward += self.food_reward
            if len(self.occupied) == self.grid_size * self.grid_size:
                done = True
                reason = "board_filled"
            else:
                self.food = self._sample_food()
            self.visit_counts = {new_head: 1}
            self.wall_follow_steps = 0
        elif opponent_gets_food and not player_gets_food:
            self.opponent_score += 1
            reward += self.opponent_food_penalty
            old_tail = self.snake.pop()
            if old_tail != new_head:
                self.occupied.discard(old_tail)
            self.food = self._sample_food()
        else:
            old_tail = self.snake.pop()
            if old_tail != new_head:
                self.occupied.discard(old_tail)

            after_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            if after_dist < before_dist:
                reward += self.distance_reward_toward
            elif after_dist > before_dist:
                reward += self.distance_penalty_away
            else:
                reward += self.stagnation_penalty

            count = self.visit_counts.get(new_head, 0) + 1
            self.visit_counts[new_head] = count
            if count >= self.loop_visit_threshold:
                reward += self.loop_visit_penalty

            if self.wall_follow_steps >= self.wall_follow_threshold:
                reward += self.wall_follow_penalty

            reward -= self.idle_step_coeff * float(self.no_food_steps)
            if self.hunger_exp_base > 0.0:
                exp_penalty = self.hunger_exp_base * (self.hunger_exp_gamma ** float(self.no_food_steps))
                reward -= min(self.hunger_exp_max, exp_penalty)

        if self.steps >= self.max_steps_factor * max(1, len(self.snake)):
            done = True
            truncated = True
            reason = "step_limit"
            reward += self.step_limit_penalty

        starvation_limit = self.starvation_steps_factor * max(1, len(self.snake))
        if not done and self.no_food_steps >= starvation_limit:
            done = True
            truncated = True
            reason = "starvation"
            reward += self.starvation_penalty

        if done:
            reward = self._apply_terminal_outcome_reward(reward, reason)

        if not done:
            self._move_blocks()
            self._try_move_food()

        return self._observation(), reward, done, truncated, self._info(reason)

    def _sample_food(self) -> tuple[int, int]:
        if self.food_spawn_radius is not None and len(self.snake) > 0:
            hr, hc = self.snake[0]
            candidates = []
            r = max(1, int(self.food_spawn_radius))
            for rr in range(max(0, hr - r), min(self.grid_size, hr + r + 1)):
                for cc in range(max(0, hc - r), min(self.grid_size, hc + r + 1)):
                    if abs(rr - hr) + abs(cc - hc) <= r and (rr, cc) not in self.occupied:
                        if (rr, cc) in self.blocks:
                            continue
                        candidates.append((rr, cc))
            if candidates:
                idx = int(self.rng.integers(0, len(candidates)))
                return candidates[idx]

        while True:
            pos = (
                int(self.rng.integers(0, self.grid_size)),
                int(self.rng.integers(0, self.grid_size)),
            )
            if pos not in self.occupied:
                if pos in self.blocks:
                    continue
                return pos

    def _observation(self, who: str = "player") -> np.ndarray:
        obs = np.zeros((self.grid_size, self.grid_size, 5), dtype=np.float32)
        if who == "player":
            head_r, head_c = self.snake[0]
            obs[head_r, head_c, 0] = 1.0
            for r, c in list(self.snake)[1:]:
                obs[r, c, 1] = 1.0
            obs[self.food[0], self.food[1], 2] = 1.0
            if self.has_opponent:
                obs[self.opponent_pos[0], self.opponent_pos[1], 3] = 1.0
            for r, c in self.blocks:
                obs[r, c, 4] = 1.0
        else:
            obs[self.opponent_pos[0], self.opponent_pos[1], 0] = 1.0
            for r, c in self.occupied:
                obs[r, c, 1] = 1.0
            obs[self.food[0], self.food[1], 2] = 1.0
            head_r, head_c = self.snake[0]
            obs[head_r, head_c, 3] = 1.0
            for r, c in self.blocks:
                obs[r, c, 4] = 1.0
        return obs

    def aux_features(self, who: str = "player") -> np.ndarray:
        if who == "player":
            head_r, head_c = self.snake[0]
            d_idx_self = self.direction_idx
        else:
            head_r, head_c = self.opponent_pos
            d_idx_self = self.opponent_dir_idx

        front_idx = d_idx_self
        right_idx = (d_idx_self + 1) % 4
        left_idx = (d_idx_self - 1) % 4

        def is_danger(dir_idx: int) -> float:
            dr, dc = self._DIRS[dir_idx]
            nr, nc = head_r + dr, head_c + dc
            if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
                return 1.0
            if who == "player":
                tail = self.snake[-1]
                if (nr, nc) in self.occupied and (nr, nc) != tail:
                    return 1.0
                if self.has_opponent and (nr, nc) == self.opponent_pos:
                    return 1.0
                if (nr, nc) in self.blocks:
                    return 1.0
            else:
                if (nr, nc) in self.occupied:
                    return 1.0
                if (nr, nc) in self.blocks:
                    return 1.0
            return 0.0

        fr, fc = self.food
        food_up = 1.0 if fr < head_r else 0.0
        food_down = 1.0 if fr > head_r else 0.0
        food_right = 1.0 if fc > head_c else 0.0
        food_left = 1.0 if fc < head_c else 0.0

        dirs = [0.0, 0.0, 0.0, 0.0]
        dirs[d_idx_self] = 1.0

        return np.array(
            [
                is_danger(front_idx),
                is_danger(right_idx),
                is_danger(left_idx),
                food_up,
                food_down,
                food_right,
                food_left,
                dirs[0],
                dirs[1],
                dirs[2],
                dirs[3],
            ],
            dtype=np.float32,
        )

    def opponent_observation(self) -> np.ndarray:
        return self._observation(who="opponent")

    def opponent_aux_features(self) -> np.ndarray:
        return self.aux_features(who="opponent")

    def set_curriculum(
        self,
        *,
        max_steps_factor: int | None = None,
        distance_reward_toward: float | None = None,
        distance_penalty_away: float | None = None,
        stagnation_penalty: float | None = None,
        loop_visit_penalty: float | None = None,
        step_limit_penalty: float | None = None,
        starvation_steps_factor: int | None = None,
        starvation_penalty: float | None = None,
        wall_follow_threshold: int | None = None,
        wall_follow_penalty: float | None = None,
        obstacle_count: int | None = None,
        moving_obstacles: bool | None = None,
        obstacle_move_period: int | None = None,
        moving_food: bool | None = None,
        food_move_prob: float | None = None,
        opponent_mode: str | None = None,
        food_spawn_radius: int | None = None,
        opponent_food_penalty: float | None = None,
        opponent_random_prob: float | None = None,
        terminal_win_reward: float | None = None,
        terminal_loss_penalty: float | None = None,
    ) -> None:
        if max_steps_factor is not None:
            self.max_steps_factor = int(max_steps_factor)
        if distance_reward_toward is not None:
            self.distance_reward_toward = float(distance_reward_toward)
        if distance_penalty_away is not None:
            self.distance_penalty_away = float(distance_penalty_away)
        if stagnation_penalty is not None:
            self.stagnation_penalty = float(stagnation_penalty)
        if loop_visit_penalty is not None:
            self.loop_visit_penalty = float(loop_visit_penalty)
        if step_limit_penalty is not None:
            self.step_limit_penalty = float(step_limit_penalty)
        if starvation_steps_factor is not None:
            self.starvation_steps_factor = max(1, int(starvation_steps_factor))
        if starvation_penalty is not None:
            self.starvation_penalty = float(starvation_penalty)
        if wall_follow_threshold is not None:
            self.wall_follow_threshold = max(1, int(wall_follow_threshold))
        if wall_follow_penalty is not None:
            self.wall_follow_penalty = float(wall_follow_penalty)
        if obstacle_count is not None:
            self.obstacle_count = max(0, int(obstacle_count))
            self.blocks = self._sample_blocks(self.obstacle_count)
        if moving_obstacles is not None:
            self.moving_obstacles = bool(moving_obstacles)
        if obstacle_move_period is not None:
            self.obstacle_move_period = max(1, int(obstacle_move_period))
        if moving_food is not None:
            self.moving_food = bool(moving_food)
        if food_move_prob is not None:
            self.food_move_prob = float(np.clip(food_move_prob, 0.0, 1.0))
        if opponent_mode is not None:
            self.opponent_mode = str(opponent_mode)
        if opponent_food_penalty is not None:
            self.opponent_food_penalty = float(opponent_food_penalty)
        if opponent_random_prob is not None:
            self.opponent_random_prob = float(np.clip(opponent_random_prob, 0.0, 1.0))
        if terminal_win_reward is not None:
            self.terminal_win_reward = float(terminal_win_reward)
        if terminal_loss_penalty is not None:
            self.terminal_loss_penalty = float(terminal_loss_penalty)
        self.food_spawn_radius = None if food_spawn_radius is None else int(food_spawn_radius)

    def _info(self, reason: str) -> dict:
        info = StepInfo(
            score=self.score,
            opponent_score=self.opponent_score,
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
        rival = (80, 120, 240)
        block = (190, 190, 190)

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

        if self.has_opponent:
            rr, rc = self.opponent_pos
            orect = self._pygame.Rect(
                rc * self.cell_pixels + 3,
                rr * self.cell_pixels + 3,
                self.cell_pixels - 6,
                self.cell_pixels - 6,
            )
            self._pygame.draw.rect(self._screen, rival, orect, border_radius=4)

        for br, bc in self.blocks:
            brect = self._pygame.Rect(
                bc * self.cell_pixels + 5,
                br * self.cell_pixels + 5,
                self.cell_pixels - 10,
                self.cell_pixels - 10,
            )
            self._pygame.draw.rect(self._screen, block, brect, border_radius=3)

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
