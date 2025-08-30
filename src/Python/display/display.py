import sys

import pygame

from Python.learn_2_slither import Learn2Slither

from ..constants import GAME_SPEED


class Display:
    _instance = None

    def __init__(self):
        if Display._instance is not None:
            raise RuntimeError(
                "Use Display.get_instance() instead of instantiating directly"
            )
        pygame.init()
        self.font = pygame.font.SysFont("Arial", 28)
        self.screen = pygame.display.set_mode((1024, 768))
        self.clock = pygame.time.Clock()
        self.running = True
        self.clock_tick = 30
        self.human_speed = True

    @classmethod
    def get_instance(cls):
        """Return the shared Display instance, create if needed."""
        if cls._instance is None:
            cls._instance = Display()
        return cls._instance

    def draw_text(self, text, x, y, color=(255, 255, 255)):
        label = self.font.render(text, True, color)
        self.screen.blit(label, (x, y))

    def fill(self, color=(0, 0, 0)):
        self.screen.fill(color)

    def flip(self):
        pygame.display.flip()

    def display_menu(self, text, i, color):
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
        self.screen.fill((0, 0, 0))
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

        text = self.font.render(
            f"Size: {len(self.main_game.snake)}", True, (255, 255, 255)
        )
        self.screen.blit(
            text,
            (self.main_game.width * self.cell_size - text.get_width() - 5, 5),
        )
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
