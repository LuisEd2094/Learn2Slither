import os
from argparse import Namespace

import pygame
from snake_ai import choose_direction

from ..constants import CELL_SIZE
from ..snake_game import Direction, SnakeGame


class Learn2Slither:
    def __init__(self, args: Namespace):
        self.sessions: int = args.sessions
        if self.sessions < 1:
            raise ValueError("Number of sessions must be at least 1.")
        # TODO check valid paths
        self.save_path: str = args.save_path
        self.load_path: str = args.load_path
        if self.save_path and not os.path.isdir(os.path.dirname(self.save_path)):
            raise ValueError(f"Save path directory does not exist: {self.save_path}")

        if self.load_path and not os.path.isfile(self.load_path):
            raise ValueError(f"Load path does not exist: {self.load_path}")

        self.learn: bool = args.learn
        self.human_speed: bool = args.human_speed
        self.pve: bool = args.pve
        self.grid_size: int = args.grid_size

        # Initialize main game, when in PVE mode we main game will be used
        # by the player and secondary game will be used by the AI
        # When NOT in pve mode, then it's only the AI that will use it.
        self.main_game = SnakeGame(width=self.grid_size, height=self.grid_size)
        self.visuals: bool = args.visuals
        self.screen = None
        self.clock = None
        self.cell_size = CELL_SIZE

        if self.pve:
            self.secondary_game = SnakeGame(width=self.grid_size, height=self.grid_size)
            self.human_speed = True
            self.visuals = True
        if self.visuals:
            self._init_pygame()

    def run(self):
        self._run_game()

    def _init_pygame(self):
        pygame.init()
        game = self.main_game
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

    def _stop_pygame(self):
        if self.visuals:
            pygame.quit()

    def _render_game(self):
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

    def _run_game(self):
        current_run = 0
        while current_run < self.sessions:
            self._set_next_move_ai(self.main_game)
            self.main_game.step()
            self.main_game.render_terminal()
            self._render_game()
            if self.human_speed:
                self.clock.tick(5)

            if self.main_game.game_over:
                current_run += 1
                self.main_game.reset()

        self._stop_pygame

    def _get_direction_ai(self, snake_view):
        dx, dy = choose_direction(snake_view)
        for d in Direction:
            if d.value == (dx, dy):
                direction = d
                break

        return direction

    def _set_next_move_ai(self, game):
        snake_view = game.get_snake_view()
        direction = self._get_direction_ai(snake_view)
        game.set_direction(direction)

    """     def _play_game_user(
        self,
        game,
        use_box=True,
    ):
        # TODO, remove this function once is no longer useful as reference for a human game
        pygame.init()
        cell_size = 30
        # Get copy of initial game state
        game = game
        # PYGAME needs a display to capture the keyboard inputs
        screen = (
            pygame.display.set_mode((1, 1))
            if not use_box
            else pygame.display.set_mode(
                (game.width * cell_size, game.height * cell_size)
            )
        )
        clock = pygame.time.Clock()

        running = True
        direction_map = {
            pygame.K_UP: Direction.UP,
            pygame.K_DOWN: Direction.DOWN,
            pygame.K_LEFT: Direction.LEFT,
            pygame.K_RIGHT: Direction.RIGHT,
        }

        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in direction_map:
                        game.set_direction(direction_map[event.key])
                    elif event.key == pygame.K_r:
                        game.reset()
                    elif event.key == pygame.K_q:
                        running = False

            # Move snake
            if not game.game_over:
                game.step()

            if use_box:
                # Draw full grid
                screen.fill((0, 0, 0))
                grid = game.get_state()
                for y in range(game.height):
                    for x in range(game.width):
                        rect = pygame.Rect(
                            x * cell_size, y * cell_size, cell_size, cell_size
                        )
                        if grid[y][x] == 1:
                            pygame.draw.rect(screen, (0, 200, 0), rect)
                        elif grid[y][x] == 2:
                            pygame.draw.rect(screen, (0, 255, 0), rect)
                        elif grid[y][x] == 3:
                            pygame.draw.rect(screen, (255, 0, 0), rect)
                        pygame.draw.rect(screen, (50, 50, 50), rect, 1)
                    font = pygame.font.SysFont(None, 24)
                text = font.render(
                    f"Size: {len(game.snake)}", True, (255, 255, 255)
                )
                screen.blit(
                    text, (game.width * cell_size - text.get_width() - 5, 5)
                )
                pygame.display.flip()

            game.render_terminal()
            clock.tick(5)
        pygame.quit() """
