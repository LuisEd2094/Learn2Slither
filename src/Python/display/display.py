import sys

import pygame

from Python.constants import (
    BACKGROUND_TILE,
    GAME_SPEED,
    LIGHT_BLUE,
    LIGHT_GREEN,
    YELLOW_ORANGE,
)
from Python.learn_2_slither import Learn2Slither


class Display:
    _instance = None

    def __init__(self):
        if Display._instance is not None:
            raise RuntimeError(
                "Use Display.get_instance() instead of instantiating directly"
            )
        pygame.init()
        self.font = pygame.font.Font(
            "/home/luis/proyects/Learn2Slither/src/assets/fonts/PressStart2P-Regular.ttf",
            28,
        )
        self.screen = pygame.display.set_mode((1024, 768))
        self.clock = pygame.time.Clock()
        self.running = True
        self.clock_tick = 30
        self.human_speed = True
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
        # Render the text in white (for mask)
        text_surf = self.font.render(text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(topleft=(x, y))

        # Create a gradient surface same size as the text
        gradient = pygame.Surface(text_surf.get_size(), pygame.SRCALPHA)
        height = text_surf.get_height()

        # Linear interpolation between colors
        for y_pos in range(height):
            # Calculate which two colors to interpolate
            total_segments = len(colors) - 1
            segment_height = height / total_segments
            segment_index = int(y_pos // segment_height)
            if segment_index >= total_segments:
                segment_index = total_segments - 1

            c1 = colors[segment_index]
            c2 = colors[segment_index + 1]

            # Interpolation factor (0-1)
            factor = (y_pos - segment_index * segment_height) / segment_height
            r = int(c1[0] + (c2[0] - c1[0]) * factor)
            g = int(c1[1] + (c2[1] - c1[1]) * factor)
            b = int(c1[2] + (c2[2] - c1[2]) * factor)

            pygame.draw.line(
                gradient, (r, g, b), (0, y_pos), (text_surf.get_width(), y_pos)
            )

        # Apply the text as a mask to the gradient
        gradient.blit(text_surf, (0, 0), None, pygame.BLEND_RGBA_MULT)

        # Blit final gradient text to the screen
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
        self.clock_tick = GAME_SPEED
        self.pve = l2s.pve
        self.l2s = l2s

        # compute cell size so grid fits screen
        screen_w, screen_h = self.screen.get_size()
        self.cell_size = min(
            screen_w // self.main_game.width,
            screen_h // self.main_game.height,
        )
        self.offset_x = (screen_w - self.main_game.width * self.cell_size) // 2
        self.offset_y = (screen_h - self.main_game.height * self.cell_size) // 2

    def render_game(self):
        self.screen.blit(self.background, (0, 0))
        grid = self.main_game.get_state()
        for y in range(self.main_game.height):
            for x in range(self.main_game.width):
                rect = pygame.Rect(
                    self.offset_x + x * self.cell_size,
                    self.offset_y + y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                if grid[y][x] == 1:
                    pygame.draw.rect(self.screen, (0, 200, 0), rect)
                elif grid[y][x] == 2:
                    pygame.draw.rect(self.screen, (0, 255, 0), rect)
                elif grid[y][x] == 3:
                    pygame.draw.rect(self.screen, (255, 0, 0), rect)
                pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)

        text = f"Size: {len(self.main_game.snake)}"
        self.draw_text(text, 5, 5, YELLOW_ORANGE)
        pygame.display.flip()

    def render_menu(self):
        pass

    def tick(self):
        if not self.human_speed:
            return
        self.clock.tick(self.clock_tick)

    def quit(self):
        pygame.quit()
        sys.exit()
