import os
from argparse import Namespace

import numpy as np
import pygame

from Python.constants.constants import (
    GAME_GRID_SIZE,
    MAX_STEPS_PER_EPISODE,
    REWARD_DEATH,
    REWARD_FOOD_EATEN,
    REWARD_NEUTRAL,
    REWARD_RED_APPLE_EATEN,
    STATE_SIZE,
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

        self.last_heading = self.main_game.get_heading()
        # Apple memory (relative direction last seen)
        self.mem_green_dirs: set[str] = set()
        self.mem_red_dir: str | None = None
        self.episode_age = 0

        # Initialize DQN agent
        self.agent = DQNAgent()
        if self.save_path:
            save_dir = os.path.dirname(os.path.abspath(self.save_path))
            if not os.path.isdir(save_dir):
                raise ValueError(f"Save path directory does not exist: {save_dir}")
        if self.load_path:
            if not os.path.isfile(os.path.abspath(self.load_path)):
                raise ValueError(f"Load path is not a file: {self.load_path}")
            self._load_contents()

    def _save_contents(self):
        """Save the trained DQN agent model to disk.

        Uses the save_path specified in __init__.
        """
        self.agent.save(self.save_path)
        print(f"Neural network model saved to {self.save_path}")

    def _load_contents(self):
        """Load a trained DQN agent model from disk.

        Uses the load_path specified in __init__.
        """
        self.agent.load(self.load_path)
        print(f"Neural network model loaded from {self.load_path}")

    def run(self):
        """Start the training or gameplay loop.

        Dispatches to either AI-only training or PvE mode based on configuration.
        """
        if not self.pve:
            self._run_only_ai()
        else:
            self._run_pve()

    def _stop_pygame(self):
        """Clean up pygame and save model if specified.

        Called at the end of training/gameplay to ensure proper cleanup.
        """
        if self.save_path:
            self._save_contents()
        if self.visuals:
            self.display.quit()

    def _run_pve(self):
        """Run Player vs Environment (human vs AI) mode.

        Runs the game loop with both human player and AI agent.
        Human controls secondary_game, AI controls main_game.
        """
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

    def _play_game_user(self, game: SnakeGame):
        """Handle user input and execute one step of the user-controlled game.

        Processes keyboard events (arrow keys to move, R to reset, Q to quit).
        Returns game_over and reset flags.

        Parameters
        ----------
        game : SnakeGame
            The game instance controlled by the user.

        Returns
        -------
        tuple
            (game_over, reset) - whether game ended or reset was requested
        """
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

        game.step()

        return (game_over or game.get_game_over()), reset

    def _play_move_ai(self, game: SnakeGame):
        """Execute one step of the AI-controlled game.

        Gets the next move from the agent and executes it.

        Parameters
        ----------
        game : SnakeGame
            The game instance controlled by the AI.

        Returns
        -------
        bool
            Whether the game ended.
        """
        self._set_next_move_ai(game)
        game.step()

        return game.get_game_over()

    def _get_reward(self, prev_len, done):
        """Centralized reward calculation - event based only."""
        if done:
            reward = REWARD_DEATH
        else:
            new_len = self.main_game.get_snake_len()

            if new_len < prev_len:
                reward = REWARD_RED_APPLE_EATEN
            elif new_len > prev_len:
                reward = REWARD_FOOD_EATEN
            else:
                reward = REWARD_NEUTRAL
        return reward

    def _get_next_action(self, game):
        """Get the next action from the agent.

        Parameters
        ----------
        game : SnakeGame
            The game instance.

        Returns
        -------
        tuple
            (state_features, action_idx, direction)
        """
        heading = game.get_heading()
        state = self.agent.get_state(game)
        action_idx = self.agent.get_action(state)
        direction = self._action_idx_to_direction(heading, action_idx)
        return state, action_idx, direction

    def _process_step_training(
        self,
        prev_features,
        action_idx,
        reward,
        new_features,
        done,
        prev_len,
        new_length,
    ):
        """Process training for a single step (short memory).

        Stores experience in memory and trains on it immediately (short-term learning).
        Updates state references for the next iteration if episode continues.

        Parameters
        ----------
        prev_features : np.ndarray
            State features before action.
        action_idx : int
            Action taken.
        reward : float
            Reward received.
        new_features : np.ndarray
            State features after action.
        done : bool
            Whether episode ended.
        prev_len : int
            Previous snake length.
        new_length : int
            Current snake length.

        Returns
        -------
        tuple
            (updated_prev_features, updated_prev_len)
        """
        self.agent.remember(prev_features, action_idx, reward, new_features, done)
        self.agent.train_short_memory(
            prev_features, action_idx, reward, new_features, done
        )

        if not done:
            return new_features, new_length
        return prev_features, prev_len

    def _handle_episode_completion(self, current_run, max_snake_len, number_of_steps):
        """Handle episode completion: logging, reset, and long memory training.

        Parameters
        ----------
        current_run : int
            Current session number (1-indexed).
        max_snake_len : int
            Maximum snake length achieved in episode.
        number_of_steps : int
            Total steps taken across all episodes.
        """
        self.agent.n_games += 1

        self.main_game.reset()
        self.last_heading = self.main_game.get_heading()
        self.mem_green_dirs = set()
        self.mem_red_dir = None

        if self.learn:
            self.agent.train_long_memory()

        print(f"Session {current_run}/{self.sessions} completed.")
        print(f"Epsilon: {self.agent.epsilon:.4f}")
        print(f"Loss: {self.agent.last_loss:.4f}")
        print(f"Learning Rate: {self.agent.current_lr:.6f}")
        print(f"Number of steps: {number_of_steps}")
        print(f"Max snake length: {max_snake_len}")
        if self.learn:
            print(f"Memory size: {len(self.agent.memory)}")
        try:
            self.plotter.update(
                max_snake_len,
                number_of_steps,
                self.agent.last_loss,
                self.agent.current_lr,
            )
        except Exception:
            pass

    def _render_if_needed(self):
        """Render game and handle display timing if enabled."""
        if self.visuals:
            self.main_game.render_terminal()
            self.display.render_game()
        if self.human_speed:
            self.display.tick()

    def _get_new_state(self, done):
        """Get the new state after taking a step.

        If the episode is not done, retrieves the actual new state from the game.
        If done, returns a zero state (terminal state).

        Parameters
        ----------
        done : bool
            Whether the episode ended.

        Returns
        -------
        tuple
            (new_features, new_length, new_heading)
        """
        if not done:
            new_length = self.main_game.get_snake_len()
            new_features = self.agent.get_state(self.main_game)
        else:
            new_length = self.main_game.get_snake_len()
            new_features = np.zeros(STATE_SIZE, dtype=np.float32)

        return new_features, new_length

    def _train_step(
        self, prev_features, action_idx, done, prev_len, current_steps, max_steps
    ):
        """Execute a single training step.

        Handles reward calculation, state retrieval, short-term training,
        and timeout checking.

        Parameters
        ----------
        prev_features : np.ndarray
            Previous state features.
        action_idx : int
            Action taken.
        done : bool
            Whether episode ended.
        prev_len : int
            Previous snake length.
        current_steps : int
            Current step count in episode.
        max_steps : int
            Maximum steps allowed per episode.

        Returns
        -------
        tuple
            (updated_prev_features, updated_prev_len, new_current_steps, should_end_episode)
        """
        current_steps += 1
        self.episode_age += 1
        reward = self._get_reward(prev_len, done)
        new_features, new_length = self._get_new_state(done)
        prev_features, prev_len = self._process_step_training(
            prev_features, action_idx, reward, new_features, done, prev_len, new_length
        )
        should_end = current_steps >= max_steps
        return prev_features, prev_len, current_steps, should_end

    def _run_only_ai(self):
        """Run AI-only training loop.

        Main training loop that runs the agent for multiple sessions,
        with optional learning, visualization, and performance tracking.
        """
        current_run = 0
        number_of_steps = 0
        max_snake_len = 0
        max_steps = MAX_STEPS_PER_EPISODE
        current_steps = 0

        while current_run < self.sessions:
            pygame.event.get()  # Prevent window from becoming unresponsive
            self.main_game.get_snake_view()
            prev_len = self.main_game.get_snake_len()

            prev_features, action_idx, direction = self._get_next_action(self.main_game)
            self.main_game.set_direction(direction)
            self.main_game.step()

            done = self.main_game.get_game_over()
            current_len = self.main_game.get_snake_len()
            max_snake_len = max(max_snake_len, current_len)

            if self.learn:
                prev_features, prev_len, current_steps, should_end = self._train_step(
                    prev_features, action_idx, done, prev_len, current_steps, max_steps
                )

                if should_end:
                    self.main_game.game_over = True

            self._render_if_needed()
            number_of_steps += 1

            if self.main_game.game_over:
                current_run += 1
                self._handle_episode_completion(
                    current_run, max_snake_len, number_of_steps
                )
                self.episode_age = 0
                max_snake_len = 0
                number_of_steps = 0
                current_steps = 0

        self._stop_pygame()

    def _action_idx_to_direction(self, heading, action_idx):
        """Convert agent action index to game direction relative to current heading.

        Maps action indices to directions relative to the snake's current heading:
        - Action 0: move straight (forward)
        - Action 1: turn left
        - Action 2: turn right

        Parameters
        ----------
        heading : tuple
            Current direction vector (dx, dy).
        action_idx : int
            Action index (0, 1, or 2).

        Returns
        -------
        Direction
            The absolute direction to move.
        """
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
        """Get the AI's next move and set it on the game.

        Retrieves state, gets action from agent, converts to direction, and sets it.

        Parameters
        ----------
        game : SnakeGame
            The game instance to set the direction on.

        Returns
        -------
        int
            The action index chosen by the agent.
        """
        features = self.agent.get_state(game)
        heading = game.get_heading()

        action_idx = self.agent.get_action(features)
        direction = self._action_idx_to_direction(heading, action_idx)
        game.set_direction(direction)
        return action_idx
