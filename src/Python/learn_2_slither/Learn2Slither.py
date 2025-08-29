import os
from argparse import Namespace

import pygame
import snake_ai

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
        if self.human_speed and not self.visuals:
            self.clock = pygame.time.Clock()
        self.dx, self.dy = 0.0, 0.0
        self.max_possible_length = self.main_game.width * self.main_game.height
        snake_ai.init(
            alpha=0.15,
            gamma=0.97,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
        )

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

    def _learn(
        self,
        prev_state,
        action,
        reward,
        s_next,
    ):
        # Implement learning logic here

        snake_ai.learn(
            prev_state,
            action,
            reward,
            s_next,
        )

    def find_green_apple(self, prev_view, head_x, head_y):
        """Return the coordinates of the first green apple seen, or None."""
        h, w = len(prev_view), len(prev_view[0])
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dx, dy in directions:
            x, y = head_x, head_y
            while 0 <= x < h and 0 <= y < w:
                cell = prev_view[x][y]
                if cell not in [" ", "."]:
                    if cell == "G":
                        return (x, y)
                    break
                x += dx
                y += dy

        return None

    def calculate_based_on_move_to_apple(self, prev_view, prev_head_x, prev_head_y):
        apple_pos_prev = self.find_green_apple(prev_view, prev_head_x, prev_head_y)
        reward = 0.0
        if apple_pos_prev is not None:
            head_x, head_y = self.main_game.get_snake_head()
            prev_distance = abs(prev_head_x - apple_pos_prev[0]) + abs(
                prev_head_y - apple_pos_prev[1]
            )
            new_distance = abs(head_x - apple_pos_prev[0]) + abs(
                head_y - apple_pos_prev[1]
            )

            if new_distance < prev_distance:
                reward += 0.1
            elif new_distance > prev_distance:
                reward -= 0.05
        return reward

    def _get_reward(self, prev_len, done, prev_head_x, prev_head_y, prev_view):
        # Base reward for surviving, negative to avoid stalling
        reward = -0.01

        if done:
            reward = -1.0
        else:
            # Change in snake length
            snake_len = self.main_game.get_snake_len()
            delta_len = snake_len - prev_len

            if delta_len > 0:
                # Reward grows with snake length
                reward += 0.5
            elif delta_len < 0:
                # Penalty smaller if snake is big, bigger if small
                reward -= 0.2
            else:
                reward += self.calculate_based_on_move_to_apple(
                    prev_view, prev_head_x, prev_head_y
                )

        # Keep it within -1.0 and 1.0
        reward = max(-1.0, min(1.0, reward))

        return reward

    def _run_game(self):
        current_run = 0
        number_of_steps = 0
        max_snake_len = 0
        while current_run < self.sessions:
            # Get previous state information before AI moves
            if self.learn:
                prev_view = self.main_game.get_snake_view()
                prev_heading = self.main_game.get_heading()
                prev_snake_head = self.main_game.get_snake_head()
                prev_len = self.main_game.get_snake_len()
                prev_head_x, prev_head_y = self.main_game.get_snake_head()
                prev_state = snake_ai.get_state(
                    prev_view, prev_heading, prev_snake_head
                )

            # Set next move for the AI in the game
            action = self._set_next_move_ai(self.main_game)

            # Run the next move
            self.main_game.step()

            done = self.main_game.get_done()

            max_snake_len = max(max_snake_len, self.main_game.get_snake_len())

            if self.learn:
                reward = self._get_reward(
                    prev_len, done, prev_head_x, prev_head_y, prev_view
                )
                new_view = self.main_game.get_snake_view() if not done else None
                new_heading = self.main_game.get_heading() if not done else None
                new_snake_head = self.main_game.get_snake_head() if not done else None
                new_state = (
                    snake_ai.get_state(new_view, new_heading, new_snake_head)
                    if not done
                    else None
                )

                self._learn(
                    prev_state=prev_state,
                    action=action,
                    reward=reward,
                    s_next=new_state,
                )
            if self.visuals:
                self.main_game.render_terminal()
                self._render_game()
            if self.human_speed:
                self.clock.tick(5)
            number_of_steps += 1
            if self.main_game.game_over:
                current_run += 1
                self.main_game.reset()
                print(f"Session {current_run}/{self.sessions} completed.")
                print("Number of states:", snake_ai.get_numstates())
                print("Number of steps:", number_of_steps)
                print("Max snake length:", max_snake_len)
                max_snake_len = 0
                number_of_steps = 0

        self._stop_pygame()

    def _get_direction_ai(self, state, heading):
        self.dx, self.dy = snake_ai.act(state, heading)
        for d in Direction:
            if d.value == (self.dx, self.dy):
                direction = d
                break

        action = snake_ai.get_action(heading, (self.dx, self.dy))

        return direction, action

    def _set_next_move_ai(self, game):
        snake_view = game.get_snake_view()
        state = snake_ai.get_state(
            snake_view, game.get_heading(), game.get_snake_head()
        )
        direction, action = self._get_direction_ai(state, game.get_heading())
        game.set_direction(direction)

        return action

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
