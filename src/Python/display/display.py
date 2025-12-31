import sys

import pygame

from Python.constants import (
    BACKGROUND_TILE,
    FONT,
    GAME_SPEED,
    LIGHT_BLUE,
    LIGHT_GREEN,
    YELLOW_ORANGE,
)
from Python.learn_2_slither import Learn2Slither
from Python.snake_game import Objects, SnakeGame


class Display:
    _instance = None

    def __init__(self):
        if Display._instance is not None:
            raise RuntimeError(
                "Use Display.get_instance() instead of instantiating directly"
            )
        pygame.init()
        self.font = pygame.font.Font(
            FONT,
            28,
        )
        self.screen = pygame.display.set_mode((1024, 768))
        self.clock = pygame.time.Clock()
        self.running = True
        self.human_speed = True
        self.clock_tick = GAME_SPEED
        self._time_accumulator = 0
        self.background = self.get_background(BACKGROUND_TILE)

    def get_background(self, file):
        background_tile = pygame.image.load(file)
        screen_w, screen_h = self.screen.get_size()
        background = pygame.Surface((screen_w, screen_h))
        background.blit(background_tile, (0, 0))
        return background

    @classmethod
    def get_instance(cls):
        """Return the shared Display instance, create if needed."""
        if cls._instance is None:
            cls._instance = Display()
        return cls._instance

    def draw_text(self, text, x, y, colors=[(255, 0, 0), (255, 255, 0)]):
        """
        Draw text with a vertical gradient.
        colors: list of RGB tuples, e.g., [(255,0,0), (255,255,0)]
        """
        text_surf = self.font.render(text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(topleft=(x, y))

        gradient = pygame.Surface(text_surf.get_size(), pygame.SRCALPHA)
        height = text_surf.get_height()

        for y_pos in range(height):
            total_segments = len(colors) - 1
            segment_height = height / total_segments
            segment_index = int(y_pos // segment_height)
            if segment_index >= total_segments:
                segment_index = total_segments - 1

            c1 = colors[segment_index]
            c2 = colors[segment_index + 1]

            factor = (y_pos - segment_index * segment_height) / segment_height
            r = int(c1[0] + (c2[0] - c1[0]) * factor)
            g = int(c1[1] + (c2[1] - c1[1]) * factor)
            b = int(c1[2] + (c2[2] - c1[2]) * factor)

            pygame.draw.line(
                gradient, (r, g, b), (0, y_pos), (text_surf.get_width(), y_pos)
            )

        gradient.blit(text_surf, (0, 0), None, pygame.BLEND_RGBA_MULT)

        self.screen.blit(gradient, text_rect)

    def fill(self, color=(0, 0, 0)):
        self.screen.fill(color)

    def flip(self):
        pygame.display.flip()

    def display_menu(self, selected_index, items, options, difficulty_levels):
        self.screen.blit(self.background, (0, 0))
        for i, item in enumerate(items):
            color = LIGHT_BLUE if i == selected_index else LIGHT_GREEN

            if item == "START":
                text = ">>> START GAME <<<"
            else:
                value = options[item]
                if item == "difficulty":
                    value = difficulty_levels.get(value, "unknown")
                text = f"{item}: {value}"
            self.write_menu_option(text, i, color)

    def write_menu_option(self, text, i, color):
        self.draw_text(text, 60, 60 + i * 40, color)

    def init_game(self, l2s: Learn2Slither):
        self.human_speed = l2s.human_speed
        self.visuals = l2s.visuals
        self.main_game = l2s.main_game
        self.secondary_game = l2s.secondary_game
        self.pve = l2s.pve
        self.l2s = l2s

        if not self.pve:
            self._init_single_game_layout()
        else:
            self._init_pve_layout()

    def _init_single_game_layout(self):
        """Initialize grid layout for single-player mode."""
        screen_w, screen_h = self.screen.get_size()
        self.GAME_GRID_SIZE = min(
            screen_w // self.main_game.width,
            screen_h // self.main_game.height,
        )
        self.offset_x = (screen_w - self.main_game.width * self.GAME_GRID_SIZE) // 2
        self.offset_y = (screen_h - self.main_game.height * self.GAME_GRID_SIZE) // 2

    def _init_pve_layout(self):
        """Initialize grid layout for player vs AI mode (side-by-side)."""
        screen_w, screen_h = self.screen.get_size()
        half_w = screen_w // 2

        # left game (player)
        self.GAME_GRID_SIZE_left = min(
            half_w // self.main_game.width,
            screen_h // self.main_game.height,
        )
        self.offset_x_left = (
            half_w - self.main_game.width * self.GAME_GRID_SIZE_left
        ) // 2
        self.offset_y_left = (
            screen_h - self.main_game.height * self.GAME_GRID_SIZE_left
        ) // 2

        # right game (AI)
        self.GAME_GRID_SIZE_right = min(
            half_w // self.secondary_game.width,
            screen_h // self.secondary_game.height,
        )
        self.offset_x_right = (
            half_w
            + (half_w - self.secondary_game.width * self.GAME_GRID_SIZE_right) // 2
        )
        self.offset_y_right = (
            screen_h - self.secondary_game.height * self.GAME_GRID_SIZE_right
        ) // 2

    def _update_board(
        self, game: SnakeGame, offset_x, offset_y, GAME_GRID_SIZE, text_x
    ):
        grid = game.get_state()
        for y in range(game.height):
            for x in range(game.width):
                rect = pygame.Rect(
                    offset_x + x * GAME_GRID_SIZE,
                    offset_y + y * GAME_GRID_SIZE,
                    GAME_GRID_SIZE,
                    GAME_GRID_SIZE,
                )
                if grid[y][x] == Objects.SNAKE.value:
                    pygame.draw.rect(self.screen, (0, 200, 0), rect)
                elif grid[y][x] == Objects.GREEN_APPLE.value:
                    pygame.draw.rect(self.screen, (0, 255, 0), rect)
                elif grid[y][x] == Objects.RED_APPLE.value:
                    pygame.draw.rect(self.screen, (255, 0, 0), rect)
                pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)

        text = f"Size: {len(game.snake)}"
        self.draw_text(text, text_x, 5, YELLOW_ORANGE)

    def render_game(self):
        self.screen.blit(self.background, (0, 0))

        if not self.pve:
            self._update_board(
                self.main_game,
                self.offset_x,
                self.offset_y,
                self.GAME_GRID_SIZE,
                5,
            )

        else:
            self._update_board(
                self.main_game,
                self.offset_x_left,
                self.offset_y_left,
                self.GAME_GRID_SIZE_left,
                5,
            )
            self._update_board(
                self.secondary_game,
                self.offset_x_right,
                self.offset_y_right,
                self.GAME_GRID_SIZE_right,
                self.screen.get_width() // 2 + 5,
            )

        pygame.display.flip()

    def render_menu(self):
        pass

    def tick(self):
        if not self.human_speed:
            return
        dt = self.clock.tick(60) / 1000.0
        self._time_accumulator += dt

        steps = 0
        step_interval = 1.0 / self.clock_tick

        while self._time_accumulator >= step_interval:
            self._time_accumulator -= step_interval
            steps += 1

        return steps

    def quit(self):
        pygame.quit()
        sys.exit()
