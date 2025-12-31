import random
from enum import Enum

from Python.constants.constants import GREEN_APPLE_TO_SPAWN


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


class Objects(Enum):
    EMPTY = 0
    SNAKE = 1
    GREEN_APPLE = 2
    RED_APPLE = 3


class SnakeGame:
    """
    A grid-based Snake game environment with two types of apples:
    - Two green apples (normal growth).
    - One red apple (shrinks the snake by one segment, or ends the game if too short).

    The snake starts at a random valid position and grows/shrinks depending on
    apples consumed. The game ends if the snake collides with itself or the wall.

    Attributes
    ----------
    width : int
        Width of the board (minimum 10).
    height : int
        Height of the board (minimum 10).
    snake : list[tuple[int, int]]
        The snake body, stored as a list of (x, y) coordinates with the head first.
    direction : Direction
        Current direction of movement.
    pending_direction : Direction
        Buffered direction to apply on the next step (used to prevent instant reversals).
    green_apples : list[tuple[int, int]]
        Positions of the green apples (normal food).
    red_apple : tuple[int, int] | None
        Position of the red apple (shrinks snake).
    grid : list[list[int]]
        2D grid representing the game state:
            0 = empty
            1 = snake
            2 = green apple
            3 = red apple
    game_over : bool
        Whether the game has ended.

    Methods
    -------
    reset()
        Reset the board and place the snake in a valid random position.
    spawn_apples()
        Spawn green apples and a red apple in random empty cells.
    set_direction(new_direction)
        Queue a new direction for the snake (ignores 180-degree turns).
    step()
        Advance the grid by one tick, updating snake position, apple, and game state.
    get_state() -> list[list[int]]
        Return the raw integer grid state.
    get_snake_view() -> list[list[str]]
        Return a 2D view showing the snake's line of sight (row + column with walls).
    render_terminal()
        Print the snake's line of sight to the terminal for debugging.
    """

    def __init__(self, width=10, height=10):
        if width < 10 or height < 10:
            raise ValueError("Width and height must be at least 10.")
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        max_attempts = 100
        for _ in range(max_attempts):
            head_x = random.randint(1, self.width - 1)
            head_y = random.randint(1, self.height - 1)

            possible_dirs = []
            # Check if snake fits 3 cells in each direction
            if head_y - 2 >= 0:
                possible_dirs.append(Direction.UP)
            if head_y + 2 < self.height:
                possible_dirs.append(Direction.DOWN)
            if head_x - 2 >= 0:
                possible_dirs.append(Direction.LEFT)
            if head_x + 2 < self.width:
                possible_dirs.append(Direction.RIGHT)

            if not possible_dirs:
                continue

            self.direction = random.choice(possible_dirs)
            self.pending_direction = self.direction
            dx, dy = self.get_heading()

            snake_body = [
                (head_x, head_y),
                (head_x - dx, head_y - dy),
                (head_x - 2 * dx, head_y - 2 * dy),
            ]

            # Make sure all segments are in bounds
            if all(0 <= x < self.width and 0 <= y < self.height for x, y in snake_body):
                self.snake = snake_body
                # Initialize grid in memory
                self.grid = [
                    [Objects.EMPTY.value for _ in range(self.width)]
                    for _ in range(self.height)
                ]
                for x, y in self.snake:
                    self.grid[y][x] = Objects.SNAKE.value
                self.green_apples = []
                self.red_apple = None
                self.spawn_apples()  # spawns apples into the grid
                self.game_over = False
                return

        # Fallback: center if random fails
        mid_x, mid_y = self.width // 2, self.height // 2
        self.snake = [(mid_x, mid_y), (mid_x - 1, mid_y), (mid_x - 2, mid_y)]
        self.direction = Direction.RIGHT
        self.pending_direction = self.direction
        self.grid = [
            [Objects.EMPTY.value for _ in range(self.width)] for _ in range(self.height)
        ]
        for x, y in self.snake:
            self.grid[y][x] = Objects.SNAKE.value
        self.green_apples = []
        self.red_apple = None
        self.spawn_apples()
        self.game_over = False

    def spawn_apples(self):
        # Spawn green apples: 1 or 2 depending on config
        num_green_apples = GREEN_APPLE_TO_SPAWN
        while len(self.green_apples) < num_green_apples:
            self._spawn_green_apple()
        # ensure one red apple (if enabled)
        if self.red_apple is None:
            self.spawn_red_apple()

    def _spawn_green_apple(self):
        empty_cells = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if self.grid[y][x] == Objects.EMPTY.value
            and (x, y) != getattr(self, "red_apple", None)
            and (x, y) not in getattr(self, "green_apples", [])
        ]
        if empty_cells:
            pos = random.choice(empty_cells)
            self.green_apples.append(pos)
            self.grid[pos[1]][pos[0]] = Objects.GREEN_APPLE.value

    def spawn_red_apple(self):
        empty_cells = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if self.grid[y][x] == Objects.EMPTY.value
            and (x, y) not in getattr(self, "green_apples", [])
        ]
        if empty_cells:
            self.red_apple = random.choice(empty_cells)
            self.grid[self.red_apple[1]][self.red_apple[0]] = Objects.RED_APPLE.value
        else:
            self.red_apple = None

    def get_heading(self):
        """Return the current heading direction of the snake."""
        return self.direction.value

    def get_green_apples(self):
        """Return all green apple positions."""
        return list(self.green_apples)

    def get_snake_head(self):
        """Return the current head position of the snake."""
        return self.snake[0] if self.snake else None

    def get_snake_len(self):
        return len(self.snake)

    def get_game_over(self):
        return self.game_over

    def set_direction(self, new_direction: Direction):
        """Change snake direction, ignoring 180-degree reversals."""
        if (
            new_direction.value[0] * -1,
            new_direction.value[1] * -1,
        ) != self.direction.value:
            self.pending_direction = new_direction

    def _pop_snake(self):
        tail = self.snake.pop()
        self.grid[tail[1]][tail[0]] = Objects.EMPTY.value

    def get_new_head(self, direction):
        head_x, head_y = self.snake[0]
        dx, dy = direction.value
        return (head_x + dx, head_y + dy)

    def would_colide(self, head):
        return (head in self.snake) or not (
            0 <= head[0] < self.width and 0 <= head[1] < self.height
        )

    def step(self):
        """Advance the game one tick in the current direction."""
        if self.game_over:
            return

        # Apply pending direction
        self.direction = self.pending_direction
        new_head = self.get_new_head(self.direction)

        # Check collisions
        if self.would_colide(new_head):
            self.game_over = True
            return

        self.snake.insert(0, new_head)
        self.grid[new_head[1]][new_head[0]] = Objects.SNAKE.value
        if new_head in self.green_apples:
            # Ate one green apple: grow (no pop tail) and respawn that apple
            idx = self.green_apples.index(new_head)
            self._respawn_green_apple(idx)
        elif new_head == self.red_apple:
            if len(self.snake) > 2:
                self._pop_snake()
                self._pop_snake()
            else:
                self.game_over = True
                self._pop_snake()
            self.spawn_red_apple()
        else:
            self._pop_snake()

    def _respawn_green_apple(self, index: int):
        # Find new empty cell for this green apple index
        empty_cells = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if self.grid[y][x] == Objects.EMPTY.value
            and (x, y) != getattr(self, "red_apple", None)
            and (x, y) not in getattr(self, "green_apples", [])
        ]
        if empty_cells:
            pos = random.choice(empty_cells)
            self.green_apples[index] = pos
            self.grid[pos[1]][pos[0]] = Objects.GREEN_APPLE.value

    def get_state(self):
        """Return a grid representation:
        0=empty, 1=snake, 2=green apple, 3=red apple"""

        return self.grid

    def get_snake_view(self):
        symbols = {0: '.', 1: 'S', 2: 'G', 3: 'R'}
        head_x, head_y = self.snake[0]
        w, h = self.width, self.height

        view = [[' ' for _ in range(w + 2)] for _ in range(h + 2)]

        grid = self.get_state()
        hx, hy = head_x + 1, head_y + 1

        # Print walls and visible cells
        for y in range(head_y, -1, -1):
            view[y + 1][hx] = symbols[grid[y][head_x]]
        view[0][hx] = 'W'
        for y in range(head_y, h):
            view[y + 1][hx] = symbols[grid[y][head_x]]
        view[h + 1][hx] = 'W'
        for x in range(head_x, -1, -1):
            view[hy][x + 1] = symbols[grid[head_y][x]]
        view[hy][0] = 'W'
        for x in range(head_x, w):
            view[hy][x + 1] = symbols[grid[head_y][x]]
        view[hy][w + 1] = 'W'

        view[hy][hx] = 'S'

        return view

    def render_terminal(self):
        """
        Render the snake's full line-of-sight in four directions.
        The snake sees all cells in its row and column until the edges.
        Surrounding walls are included at the grid edges.
        """
        view = self.get_snake_view()
        print(f"Snake size: {len(self.snake)}\n")
        for row in view:
            if any(c != ' ' for c in row):
                print(' '.join(row))
        print()
