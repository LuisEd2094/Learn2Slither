import pygame

from Python.learn_2_slither import Learn2Slither

from ..constants import CELL_SIZE, GAME_SPEED


class Display:
    def __init__(self, l2s: Learn2Slither):
        self.human_speed = l2s.human_speed
        self.visuals = l2s.visuals
        self.main_game = l2s.main_game
        self.cell_size = CELL_SIZE
        self.game_speed = GAME_SPEED
        self.human_speed = l2s.human_speed
        self.screen = None
        self.pve = l2s.pve

        if self.visuals:
            pygame.init()
            self.l2s = l2s
            game = self.l2s.main_game
            # If we are playing in PVE mode, we need a larger display
            # Otherwise we can just display one screen.

            self.screen = (
                pygame.display.set_mode(
                    (game.width * self.cell_size, game.height * self.cell_size)
                )
                if not self.pve
                else pygame.display.set_mode(
                    (
                        game.width * self.cell_size * 2,
                        game.height * self.cell_size * 2,
                    )
                )
            )
            self.clock = pygame.time.Clock()
        if self.human_speed and not self.visuals:
            self.clock = pygame.time.Clock()

    def render(self):
        if not self.visuals:
            return
        self.screen.fill((0, 0, 0))
        grid = self.main_game.get_state()
        for y in range(self.main_game.height):
            for x in range(self.main_game.width):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
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
            font = pygame.font.SysFont(None, 24)
        text = font.render(f"Size: {len(self.main_game.snake)}", True, (255, 255, 255))
        self.screen.blit(
            text,
            (self.main_game.width * self.cell_size - text.get_width() - 5, 5),
        )
        pygame.display.flip()

    def tick(self):
        if not self.human_speed:
            return
        self.clock.tick(self.game_speed)

    def quit(self):
        if self.visuals:
            pygame.quit()
