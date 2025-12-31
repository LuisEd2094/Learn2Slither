import os
import statistics
from argparse import Namespace

import numpy as np
import pygame

from Python.constants.constants import (
    ACTION_SIZE,
    DQN_BATCH_SIZE,
    DQN_GAMMA,
    DQN_LEARNING_RATE,
    DQN_MEMORY_SIZE,
    EPSILON_DECAY,
    EPSILON_MIN,
    EPSILON_START,
    GAME_GRID_SIZE,
    HIDDEN_SIZE,
    MAX_STEPS_PER_EPISODE,
    REWARD_DEATH,
    REWARD_FOOD_EATEN,
    REWARD_NEUTRAL,
    REWARD_RED_APPLE_EATEN,
    STATE_SIZE,
    TARGET_NETWORK_UPDATE_FREQ,
)
from Python.display import Display
from Python.display.plotter import Plotter

from ..snake_game import Direction, SnakeGame
from .dqn import DQNAgent


class Learn2Slither:
    def __init__(self, args: Namespace):
        self.sessions: int = args.sessions
        if self.sessions < 1:
            raise ValueError("Number of sessions must be at least 1.")
        # TODO check valid paths
        self.save_path: str = args.save_path or ""
        self.load_path: str = args.load_path or ""

        self.learn: bool = args.learn
        self.human_speed: bool = args.human_speed
        self.pve: bool = args.pve
        self.grid_size: int = args.grid_size

        # Initialize main game, when in PVE mode we main game will be used
        # by the player and secondary game will be used by the AI
        # When NOT in pve mode, then it's only the AI that will use it.
        self.main_game = SnakeGame(width=self.grid_size, height=self.grid_size)
        self.visuals: bool = args.visuals
        self.display: Display = None
        self.GAME_GRID_SIZE: int = GAME_GRID_SIZE
        self.offset_x: int = 0
        self.offset_y: int = 0
        self.GAME_GRID_SIZE_left: int = GAME_GRID_SIZE
        self.offset_x_left: int = 0
        self.offset_y_left: int = 0
        self.GAME_GRID_SIZE_right: int = GAME_GRID_SIZE
        self.offset_x_right: int = 0
        self.offset_y_right: int = 0
        self.clock_tick: int = 100
        self.secondary_game = None
        if self.pve:
            self.secondary_game = SnakeGame(width=self.grid_size, height=self.grid_size)
            self.human_speed = True
            self.visuals = True

        if self.visuals:
            self.display = Display.get_instance()
            self.display.init_game(self)
        # Initialize live plotter (always on to observe progress)
        self.plotter = Plotter(title="Snake DQN Progress")

        self.dx, self.dy = 0.0, 0.0
        # Movement tracking for features
        self.head_history = []
        self.last_heading = self.main_game.get_heading()
        # Apple memory (relative direction last seen)
        self.mem_green_dirs: set[str] = set()
        self.mem_red_dir: str | None = None
        # Fitness tracking (length^2 * age)
        self.episode_age = 0
        self.best_fitness = 0

        # Initialize DQN agent
        self.agent = DQNAgent(
            state_size=STATE_SIZE,
            action_size=ACTION_SIZE,
            hidden_size=HIDDEN_SIZE,
            lr=DQN_LEARNING_RATE,
            gamma=DQN_GAMMA,
            epsilon=EPSILON_START,
            epsilon_min=EPSILON_MIN,
            epsilon_decay=EPSILON_DECAY,
            batch_size=DQN_BATCH_SIZE,
            memory_size=DQN_MEMORY_SIZE,
            target_update=TARGET_NETWORK_UPDATE_FREQ,
        )
        print("Using PyTorch DQN for Q-function")
        if self.save_path:
            save_dir = os.path.dirname(os.path.abspath(self.save_path))
            if not os.path.isdir(save_dir):
                raise ValueError(f"Save path directory does not exist: {save_dir}")
        if self.load_path:
            if not os.path.isfile(os.path.abspath(self.load_path)):
                raise ValueError(f"Load path is not a file: {self.load_path}")
            self._load_contents()

    def _save_contents(self):
        self.agent.save(self.save_path)
        print(f"Neural network model saved to {self.save_path}")

    def _load_contents(self):
        self.agent.load(self.load_path)
        print(f"Neural network model loaded from {self.load_path}")

    ###############
    # RUN METHODS #
    ###############

    def run(self):
        if not self.pve:
            self._run_only_ai()
        else:
            self._run_pve()

    def _stop_pygame(self):
        if self.save_path:
            self._save_contents()
        if self.visuals:
            self.display.quit()

    def _run_pve(self):
        game_over = False
        reset = False
        user_game = self.secondary_game
        ai_game = self.main_game
        while not game_over:
            steps = self.display.tick()
            for _ in range(steps):
                game_over, reset = self._play_game_user(user_game)
                if game_over:
                    break
                game_over = self._play_move_ai(ai_game)
                if reset:
                    user_game.reset()
                    ai_game.reset()
                    reset = False
                self.display.render_game()
        self._stop_pygame()

    ################
    # USER METHODS #
    ################

    def _play_game_user(
        self,
        game: SnakeGame,
    ):
        game_over = False
        reset = False
        direction_map = {
            pygame.K_UP: Direction.UP,
            pygame.K_DOWN: Direction.DOWN,
            pygame.K_LEFT: Direction.LEFT,
            pygame.K_RIGHT: Direction.RIGHT,
        }

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            elif event.type == pygame.KEYDOWN:
                if event.key in direction_map:
                    game.set_direction(direction_map[event.key])
                elif event.key == pygame.K_r:
                    reset = True
                elif event.key == pygame.K_q:
                    game_over = True

        # Move snake
        game.step()

        return (game_over or game.get_game_over()), reset

    ####################
    # AI LOGIC METHODS #
    ####################

    def _play_move_ai(self, game: SnakeGame):
        pygame.event.get()  # Prevent window from becoming unresponsive
        self._set_next_move_ai(game)
        game.step()

        return game.get_game_over()

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

    def _get_reward(self, prev_len, done, prev_red_apple=None):
        """Centralized reward calculation - event based only."""
        if done:
            reward = REWARD_DEATH
        else:
            new_len = self.main_game.get_snake_len()

            if new_len < prev_len:
                # Red apple was eaten (snake shrank)
                reward = REWARD_RED_APPLE_EATEN
            elif new_len > prev_len:
                # Green apple was eaten (snake grew)
                reward = REWARD_FOOD_EATEN
            else:
                # No event, neutral reward
                reward = REWARD_NEUTRAL
        return reward

    def _run_only_ai(self):
        current_run = 0
        number_of_steps = 0
        max_snake_len = 0
        max_steps = MAX_STEPS_PER_EPISODE
        current_steps = 0
        _ = True
        seen_states = set()
        frame_iteration = 0

        while current_run < self.sessions:
            # Epsilon is now calculated dynamically in get_action() based on n_games

            self.main_game.get_snake_view()
            prev_heading = self.main_game.get_heading()
            prev_snake_head = self.main_game.get_snake_head()
            # seed history if empty
            if not self.head_history or self.head_history[-1] != prev_snake_head:
                self.head_history.append(prev_snake_head)
            prev_len = self.main_game.get_snake_len()
            prev_head_x, prev_head_y = self.main_game.get_snake_head()
            frame_iteration = 0

            # Get initial state from DQN agent
            prev_features = self.agent.get_state(self.main_game)
            prev_state_val = None

            if self.learn:
                if current_run > (self.sessions // 4):
                    _ = True
                if prev_state_val is not None:
                    seen_states.add(prev_state_val)
                current_steps += 1
                self.episode_age += 1

            # Use the DQN agent to get action
            action_idx = self.agent.get_action(prev_features)
            direction = self._action_idx_to_direction(prev_heading, action_idx)
            self.main_game.set_direction(direction)
            _ = action_idx

            # Store red apple position before step
            prev_red_apple = self.main_game.red_apple

            pygame.event.get()  # Prevent window from becoming unresponsive
            self.main_game.step()
            frame_iteration += 1

            done = self.main_game.get_game_over()

            # Fallback: check frame timeout to prevent infinite episodes
            current_len = self.main_game.get_snake_len()
            if not done and frame_iteration > 100 * max(1, current_len):
                done = True

            max_snake_len = max(max_snake_len, current_len)

            if self.learn:
                reward = self._get_reward(prev_len, done, prev_red_apple)

                _ = None
                if not done:
                    _ = self.main_game.get_snake_view()
                    new_heading = self.main_game.get_heading()
                    new_snake_head = self.main_game.get_snake_head()
                    new_length = self.main_game.get_snake_len()
                    # update movement tracking
                    self.last_heading = new_heading
                    self.head_history.append(new_snake_head)
                    new_features = self.agent.get_state(self.main_game)
                else:
                    new_length = self.main_game.get_snake_len()
                    new_features = np.zeros(
                        STATE_SIZE, dtype=np.float32
                    )  # Updated from 11 to 15

                self.agent.remember(
                    prev_features,
                    action_idx,
                    reward,
                    new_features,
                    done,
                )
                # Train immediately on this step (short memory)
                self.agent.train_short_memory(
                    prev_features, action_idx, reward, new_features, done
                )

                # Update state and tracking for next iteration
                if not done:
                    prev_features = new_features
                    prev_len = new_length
                    prev_heading = new_heading

                if current_steps >= max_steps:
                    self.main_game.game_over = True
            if self.visuals:
                self.main_game.render_terminal()
                self.display.render_game()
            if self.human_speed:
                self.display.tick()
            number_of_steps += 1
            if self.main_game.game_over:
                current_run += 1

                # Increment n_games for epsilon calculation (matching nn_test.py)
                self.agent.n_games += 1

                # Compute fitness (length^2 * age)
                fitness = (max_snake_len**2) * self.episode_age
                self.best_fitness = max(self.best_fitness, fitness)

                self.main_game.reset()
                # reset movement tracking
                self.head_history = []
                self.last_heading = self.main_game.get_heading()
                # keep apple memory across episodes? reset to None to avoid stale info
                self.mem_green_dirs = set()
                self.mem_red_dir = None

                # Train on entire memory buffer at end of episode (like reference)
                if self.learn:
                    self.agent.train_long_memory()

                print(f"Session {current_run}/{self.sessions} completed.")
                print(f"Epsilon: {self.agent.epsilon:.4f}")
                print(f"Loss: {self.agent.last_loss:.4f}")
                print(f"Learning Rate: {self.agent.current_lr:.6f}")
                if self.learn:
                    print(f"Memory size: {len(self.agent.memory)}")
                print("Number of steps:", number_of_steps)
                print("Max snake length:", max_snake_len)
                print(f"Fitness: {fitness} (best: {self.best_fitness})")
                self.episode_age = 0
                # Update live plot
                try:
                    self.plotter.update(
                        max_snake_len,
                        number_of_steps,
                        self.agent.last_loss,
                        self.agent.current_lr,
                    )

                except Exception:
                    pass
                max_snake_len = 0
                number_of_steps = 0
                current_steps = 0

        self._stop_pygame()

    def _get_direction_ai(self, state, heading):
        # Return forward action (method kept for backwards compatibility)
        self.dx, self.dy = heading
        for d in Direction:
            if d.value == (self.dx, self.dy):
                direction = d
                break
        else:
            direction = Direction.UP

        action = 0  # Forward action
        return direction, action

    def _predict_next_head(self, game, direction):
        return game.get_new_head(direction)

    def _would_collide(self, game, head):
        return game.would_colide(head)

    def _action_idx_to_direction(self, heading, action_idx):
        clock = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        heading_to_idx = {d.value: i for i, d in enumerate(clock)}
        current = heading_to_idx.get(heading, 0)
        straight = clock[current]
        left = clock[(current - 1) % 4]
        right = clock[(current + 1) % 4]

        if action_idx == 0:
            return straight
        if action_idx == 1:
            return left
        return right

    def _set_next_move_ai(self, game):
        # Get state features using DQN agent
        features = self.agent.get_state(game)
        heading = game.get_heading()

        # Get action from DQN
        action_idx = self.agent.get_action(features)
        direction = self._action_idx_to_direction(heading, action_idx)
        game.set_direction(direction)
        return action_idx

    def _compute_move_index(self) -> float:
        if len(self.head_history) < 2:
            return 1.0
        last_coords = self.head_history[-10:]
        xs = [p[0] for p in last_coords]
        ys = [p[1] for p in last_coords]
        std_x = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        std_y = statistics.pstdev(ys) if len(ys) > 1 else 0.0
        return (std_x + std_y) / (2.0 * max(1, self.grid_size))

    def _compute_last_move_flags(
        self, prev_heading, current_heading
    ) -> tuple[bool, bool, bool]:
        clock = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        h2i = {d.value: i for i, d in enumerate(clock)}
        prev_i = h2i.get(prev_heading, 0)
        curr_i = h2i.get(current_heading, 0)
        last_move_straight = prev_i == curr_i
        # A left turn means current index moved -1 from previous (counter-clockwise)
        last_move_left = curr_i == (prev_i - 1) % 4
        # A right turn means current index moved +1 from previous (clockwise)
        last_move_right = curr_i == (prev_i + 1) % 4
        return last_move_straight, last_move_left, last_move_right

    def _update_apple_memory_from_state(self, value: int) -> None:
        # State encodes rays in order: forward(0..5), left(6..11), right(12..17), back(18..23)
        # Now including back direction for cross-like vision
        dirs = [("straight", 0), ("left", 6), ("right", 12), ("back", 18)]
        for name, shift in dirs:
            ray = (value >> shift) & 0b111111
            obj_type = ray & 0b111
            if obj_type == 3:  # green
                self.mem_green_dirs.add(name)
            elif obj_type == 4:  # red
                self.mem_red_dir = name

    def _compute_green_dirs_full_vision(
        self, head: tuple[int, int], heading: tuple[int, int]
    ) -> set[str]:
        # Determine which relative directions (straight/left/right) have any green apples aligned
        # Coordinate system: head=(x,y) with x as row, y as column.
        # RIGHT increases y, LEFT decreases y, DOWN increases x, UP decreases x.
        dirs = set()
        hx, hy = head
        apples = getattr(self.main_game, "green_apples", [])
        clock = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        h2i = {d.value: i for i, d in enumerate(clock)}
        curr_i = h2i.get(heading, 0)
        straight_abs = clock[curr_i]
        left_abs = clock[(curr_i - 1) % 4]
        right_abs = clock[(curr_i + 1) % 4]
        back_abs = clock[(curr_i + 2) % 4]

        for ax, ay in apples:
            abs_dir = None
            if ay == hy:  # same column → vertical alignment
                abs_dir = Direction.DOWN if ax > hx else Direction.UP
            elif ax == hx:  # same row → horizontal alignment
                abs_dir = Direction.RIGHT if ay > hy else Direction.LEFT
            else:
                continue

            if abs_dir == straight_abs:
                dirs.add("straight")
            elif abs_dir == left_abs:
                dirs.add("left")
            elif abs_dir == right_abs:
                dirs.add("right")
            elif abs_dir == back_abs:
                dirs.add("back")

        return dirs
