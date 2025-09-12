import os
from argparse import Namespace

import pygame
import snake_ai

from Python.constants import CELL_SIZE
from Python.display import Display

from ..snake_game import Direction, SnakeGame


class Learn2Slither:
    DEFAULTS = {
        "sessions": 10,
        "save_path": "",
        "load_path": "",
        "learn": True,
        "human_speed": False,
        "pve": False,
        "grid_size": CELL_SIZE,
        "visuals": True,
        "difficulty": None,
    }

    DIFFICULTY = {
        "easy": "/home/luis/proyects/Learn2Slither/src/models/pve/easy.txt",
        "normal": "/home/luis/proyects/Learn2Slither/src/models/pve/normal.txt",
        "hard": "/home/luis/proyects/Learn2Slither/src/models/pve/hard.txt",
    }

    def __init__(self, args: Namespace):
        config = {**self.DEFAULTS, **vars(args)}
        self.sessions: int = config["sessions"]
        if self.sessions < 1:
            raise ValueError("Number of sessions must be at least 1.")
        # TODO check valid paths
        self.save_path: str = config["save_path"]
        self.load_path: str = config["load_path"]

        self.learn: bool = config["learn"]
        self.human_speed: bool = config["human_speed"]
        self.pve: bool = config["pve"]
        self.grid_size: int = config["grid_size"]
        self.difficulty: str = config["difficulty"].lower()

        # Initialize main game, when in PVE mode we main game will be used
        # by the player and secondary game will be used by the AI
        # When NOT in pve mode, then it's only the AI that will use it.
        self.main_game = SnakeGame(width=self.grid_size, height=self.grid_size)
        self.visuals: bool = config["visuals"]
        self.display: Display = None
        self.cell_size: int = CELL_SIZE
        self.offset_x: int = 0
        self.offset_y: int = 0
        self.cell_size_left: int = CELL_SIZE
        self.offset_x_left: int = 0
        self.offset_y_left: int = 0
        self.cell_size_right: int = CELL_SIZE
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

        self.dx, self.dy = 0.0, 0.0
        snake_ai.init(
            alpha=0.01,
            gamma=0.95,
            epsilon=0.99,
            epsilon_min=0.1,
            epsilon_decay=0.995,
        )

        if self.save_path:
            save_dir = os.path.dirname(os.path.abspath(self.save_path))
            if not os.path.isdir(save_dir):
                raise ValueError(f"Save path directory does not exist: {save_dir}")
        if self.difficulty in self.DIFFICULTY and not self.load_path:
            self.load_path = self.DIFFICULTY[self.difficulty]
        if self.load_path:
            if not os.path.isfile(os.path.abspath(self.load_path)):
                raise ValueError(f"Load path is not a file: {self.load_path}")
            self._load_contents()

    def _save_contents(self):
        content = snake_ai.get_q_table()

        with open(self.save_path, "w", encoding="utf-8") as f:
            for state, v1, v2, v3 in content:
                f.write(f"{state},{v1},{v2},{v3}\n")

    def _load_contents(self):
        data = []
        with open(self.load_path, "r", encoding="utf-8") as f:
            for line in f:
                state, v1, v2, v3 = line.strip().split(",")
                data.append((int(state), float(v1), float(v2), float(v3)))
        snake_ai.load_q_table(data)

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
        self._set_next_move_ai(game)
        game.step()

        return game.get_game_over()

    def _learn(
        self,
        prev_state,
        action,
        decay,
        reward,
        s_next,
    ):
        # Implement learning logic here

        snake_ai.learn(
            prev_state,
            action,
            reward,
            decay,
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
                reward -= 0.02
        return reward

    def _get_reward(self, prev_len, done, prev_head_x, prev_head_y, prev_view):
        # Base reward for surviving, negative to avoid stalling

        if done:
            reward = -1.0
        else:
            # Change in snake length
            snake_len = self.main_game.get_snake_len()
            delta_len = snake_len - prev_len
            moved_to_apple = self.calculate_based_on_move_to_apple(
                prev_view, prev_head_x, prev_head_y
            )
            if delta_len > 0:
                # Reward grows with snake length
                reward = 0.75
            elif delta_len < 0:
                # Penalty smaller if snake is big, bigger if small
                reward = -0.40
            elif moved_to_apple != 0.0:
                reward = moved_to_apple
            else:
                reward = -0.1
        # Keep it within -1.0 and 1.0
        reward = max(-1.0, min(1.0, reward))

        return reward

    def _run_only_ai(self):
        current_run = 0
        number_of_steps = 0
        max_snake_len = 0
        max_steps = 1000
        current_steps = 0
        decay = True
        seen_states = set()

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
                if current_run > (self.sessions // 4):
                    decay = True
                seen_states.add(
                    prev_state.value()
                )  # if State has .value() returning u64
                # print(f"Unique states so far: {len(seen_states)}")
                current_steps += 1

            # Set next move for the AI in the game
            action = self._set_next_move_ai(self.main_game)

            # Run the next move
            self.main_game.step()

            done = self.main_game.get_game_over()

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
                    decay=decay,
                )
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
                self.main_game.reset()
                print(f"Session {current_run}/{self.sessions} completed.")
                print("Number of states:", snake_ai.get_numstates())
                print("Number of steps:", number_of_steps)
                print("Max snake length:", max_snake_len)
                max_snake_len = 0
                number_of_steps = 0
                current_steps = 0

        self._stop_pygame()

    def _get_direction_ai(self, state, heading):
        self.dx, self.dy = snake_ai.act(state, heading)
        for d in Direction:
            if d.value == (self.dx, self.dy):
                direction = d
                break

        action = snake_ai.get_action(heading, (self.dx, self.dy))
        return direction, action

    def _predict_next_head(self, game, direction):
        return game.get_new_head(direction)

    def _would_collide(self, game, head):
        return game.would_colide(head)

    def _set_next_move_ai(self, game):
        snake_view = game.get_snake_view()
        heading = game.get_heading()
        snake_head = game.get_snake_head()
        prev_state = snake_ai.get_state(snake_view, heading, snake_head)

        tried_actions = set()
        last_action = None

        while len(tried_actions) < 3:
            direction, action = self._get_direction_ai(prev_state, heading)
            last_action = action

            if action in tried_actions:
                # Already tested this action
                continue

            # Predict if this action will immediately die
            next_head = self._predict_next_head(game, direction)
            if self._would_collide(game, next_head):
                # Learn negative reward
                self._learn(
                    prev_state=prev_state,
                    action=action,
                    reward=-1.0,
                    s_next=None,
                    decay=False,
                )
                tried_actions.add(action)
                continue
            else:
                # Safe action found
                game.set_direction(direction)
                return action

        # All actions kill: just pick the last one
        game.set_direction(direction)
        return last_action
