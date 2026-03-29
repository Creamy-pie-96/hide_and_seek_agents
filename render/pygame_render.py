"""
pygame_render.py — Real-time renderer for the hide-and-seek environment.

Visual conventions:
  Tile colours are drawn first, then a scent overlay, then agents on top.

  Hiders  — green  circle (red X when caught)
  Seekers — red    circle (with vision cone drawn faintly)

  Rooms without lights: tiles drawn dark/blue-tinted
  Scent trails: orange-tinted overlay at varying opacity

HUD (top bar):
  Step counter | Phase (PREP / HUNT) | Hiders alive | Food counts
"""

import numpy as np

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

from env.world import (EMPTY, WALL, DOOR, HIDE_SPOT, LIGHT_SW,
                       FOOD, FAKE_FOOD, BARRICADE, HEAVY_OBJ, SCENT)
from env.agent import Team

# ── Colours ──────────────────────────────────────────────────────────────────
C_BG          = ( 20,  20,  30)
C_WALL        = ( 55,  55,  70)
C_FLOOR       = ( 38,  38,  52)
C_FLOOR_DARK  = ( 18,  18,  32)   # unlit room
C_DOOR        = (120,  90,  50)
C_BARRICADE   = ( 80,  50,  30)
C_HIDE_SPOT   = ( 40,  70,  60)
C_LIGHT_SW    = (220, 200,  80)
C_FOOD        = ( 80, 200,  80)
C_FAKE_FOOD   = (180, 200,  80)
C_HEAVY_OBJ   = (140,  80,  40)
C_SCENT       = (255, 140,   0)   # orange — drawn as overlay

C_HIDER       = ( 50, 220, 120)
C_HIDER_DEAD  = (200,  50,  50)
C_HIDER_STUN  = (220, 180,  50)
C_SEEKER      = (220,  60,  60)
C_SEEKER_STUN = (180,  80, 200)

C_HUD_BG      = ( 12,  12,  20)
C_HUD_TEXT    = (200, 200, 210)
C_HUD_PREP    = (255, 200,  60)
C_HUD_HUNT    = (255,  80,  80)

TILE_COLOURS = {
    WALL:      C_WALL,
    EMPTY:     C_FLOOR,
    DOOR:      C_DOOR,
    HIDE_SPOT: C_HIDE_SPOT,
    LIGHT_SW:  C_LIGHT_SW,
    FOOD:      C_FOOD,
    FAKE_FOOD: C_FAKE_FOOD,
    BARRICADE: C_BARRICADE,
    HEAVY_OBJ: C_HEAVY_OBJ,
    SCENT:     C_FLOOR,    # scent drawn as overlay separately
}


class HideSeekRenderer:
    """
    Pygame-based renderer.

    Usage:
        renderer = HideSeekRenderer(grid_h=48, grid_w=48, tile_px=14)
        renderer.init()
        ...
        renderer.draw(env.get_render_state())
        ...
        renderer.close()
    """

    HUD_HEIGHT = 42

    def __init__(self, grid_h: int = 48, grid_w: int = 48, tile_px: int = 14):
        if not HAS_PYGAME:
            raise ImportError("pygame not installed — run: pip install pygame")
        self.grid_h  = grid_h
        self.grid_w  = grid_w
        self.tile_px = tile_px
        self.win_w   = grid_w * tile_px
        self.win_h   = grid_h * tile_px + self.HUD_HEIGHT
        self.screen  = None
        self.clock   = None
        self.font_sm = None
        self.font_md = None

    def init(self) -> None:
        pygame.init()
        self.screen  = pygame.display.set_mode((self.win_w, self.win_h))
        pygame.display.set_caption("Hide & Seek — Multi-Agent RL")
        self.clock   = pygame.time.Clock()
        self.font_sm = pygame.font.SysFont("monospace", 11, bold=False)
        self.font_md = pygame.font.SysFont("monospace", 14, bold=True)

    def draw(self, state: dict, fps: int = 30) -> bool:
        """
        Draw one frame. Returns False if the window was closed.
        state comes from env.get_render_state().
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        self.screen.fill(C_BG)

        grid      = state['grid']
        scent_map = state['scent_map']
        rooms     = state['rooms']

        # Build quick room-lit lookup
        lit_rooms = {r.room_id: r.light_on for r in rooms}

        # ── Tiles ────────────────────────────────────────────────────────────
        tp = self.tile_px
        for r in range(self.grid_h):
            for c in range(self.grid_w):
                tile  = grid[r, c]
                px, py = c * tp, r * tp + self.HUD_HEIGHT

                # Determine base colour
                colour = TILE_COLOURS.get(tile, C_FLOOR)

                # Dim unlit rooms
                room_id = None
                for room in rooms:
                    if room.contains(r, c):
                        room_id = room.room_id
                        break
                if room_id is not None and not lit_rooms.get(room_id, True):
                    # Blend toward dark
                    colour = tuple(max(0, v - 22) for v in C_FLOOR_DARK)
                    if tile in (HIDE_SPOT, LIGHT_SW, FOOD, FAKE_FOOD, HEAVY_OBJ):
                        colour = tuple(int(a * 0.45) for a in TILE_COLOURS[tile])

                pygame.draw.rect(self.screen, colour, (px, py, tp, tp))

                # Scent overlay
                scent = scent_map[r, c]
                if scent > 0.05:
                    alpha = int(scent * 100)
                    surf  = pygame.Surface((tp, tp), pygame.SRCALPHA)
                    surf.fill((*C_SCENT, alpha))
                    self.screen.blit(surf, (px, py))

        # ── Grid lines (subtle) ───────────────────────────────────────────────
        line_col = (30, 30, 40)
        for r in range(0, self.grid_h + 1):
            y = r * tp + self.HUD_HEIGHT
            pygame.draw.line(self.screen, line_col, (0, y), (self.win_w, y))
        for c in range(0, self.grid_w + 1):
            x = c * tp
            pygame.draw.line(self.screen, line_col, (x, self.HUD_HEIGHT),
                             (x, self.win_h))

        # ── Agents ───────────────────────────────────────────────────────────
        for agent_data in state['agents']:
            aid, row, col, team_val, alive, food, stunned = agent_data
            cx = col * tp + tp // 2
            cy = row * tp + tp // 2 + self.HUD_HEIGHT
            radius = tp // 2 - 1

            is_hider = team_val == Team.HIDER.value

            if not alive:
                colour = C_HIDER_DEAD
                # Draw X
                pygame.draw.line(self.screen, colour,
                                 (cx - radius, cy - radius),
                                 (cx + radius, cy + radius), 2)
                pygame.draw.line(self.screen, colour,
                                 (cx + radius, cy - radius),
                                 (cx - radius, cy + radius), 2)
            else:
                if stunned:
                    colour = C_HIDER_STUN if is_hider else C_SEEKER_STUN
                else:
                    colour = C_HIDER if is_hider else C_SEEKER
                pygame.draw.circle(self.screen, colour, (cx, cy), radius)
                # Outline
                pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), radius, 1)
                # Agent ID label
                lbl = self.font_sm.render(str(aid), True, (0, 0, 0))
                self.screen.blit(lbl, (cx - lbl.get_width() // 2,
                                       cy - lbl.get_height() // 2))
                # Food dots
                for fi in range(min(food, 5)):
                    fx = cx - 4 + fi * 2
                    pygame.draw.circle(self.screen, C_FOOD, (fx, cy + radius + 2), 1)

        # ── HUD ──────────────────────────────────────────────────────────────
        pygame.draw.rect(self.screen, C_HUD_BG, (0, 0, self.win_w, self.HUD_HEIGHT))

        step      = state['step']
        max_steps = state['max_steps']
        is_prep   = state['prep']
        caught    = state['hiders_caught']

        phase_str  = "◀ PREP  " if is_prep else "▶ HUNT  "
        phase_col  = C_HUD_PREP if is_prep else C_HUD_HUNT
        hiders_str = f"Hiders alive: {3 - caught}/3"
        step_str   = f"Step {step:>4}/{max_steps}"

        # Progress bar
        bar_w = self.win_w - 20
        bar_h = 4
        bar_x, bar_y = 10, self.HUD_HEIGHT - 8
        filled = int(bar_w * step / max_steps)
        pygame.draw.rect(self.screen, (60, 60, 80), (bar_x, bar_y, bar_w, bar_h))
        bar_colour = C_HUD_PREP if is_prep else C_HUD_HUNT
        pygame.draw.rect(self.screen, bar_colour, (bar_x, bar_y, filled, bar_h))

        y_text = 6
        for text, colour in [
            (phase_str, phase_col),
            (step_str, C_HUD_TEXT),
            (f"  |  {hiders_str}", C_HUD_TEXT),
        ]:
            surf = self.font_md.render(text, True, colour)
            self.screen.blit(surf, (10, y_text))
            # Crude inline layout — advance x
            y_text += 0   # all on same row via surface blit with x offset
            break           # rewrite below properly

        x_cursor = 10
        for text, colour in [
            (phase_str, phase_col),
            (step_str, C_HUD_TEXT),
            ("   |   ", C_HUD_TEXT),
            (hiders_str, C_HIDER if caught < 3 else C_HIDER_DEAD),
        ]:
            surf = self.font_md.render(text, True, colour)
            self.screen.blit(surf, (x_cursor, 6))
            x_cursor += surf.get_width()

        pygame.display.flip()
        self.clock.tick(fps)
        return True

    def close(self) -> None:
        pygame.quit()


def run_random_demo(steps: int = 300, fps: int = 15) -> None:
    """Quick sanity check: random agents, just watch the renderer."""
    import random
    from env.hide_seek_env import HideSeekEnv

    env      = HideSeekEnv(width=36, height=36)
    renderer = HideSeekRenderer(grid_h=36, grid_w=36, tile_px=16)
    renderer.init()

    obs, _ = env.reset(seed=42)
    running = True
    for _ in range(steps):
        if not running:
            break
        actions = {i: random.randint(0, env.action_dim - 1)
                   for i in range(6)}
        obs, rew, done, info = env.step(actions)
        running = renderer.draw(env.get_render_state(), fps=fps)
        if done['__all__']:
            obs, _ = env.reset()

    renderer.close()


if __name__ == "__main__":
    run_random_demo()