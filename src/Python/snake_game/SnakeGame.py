import random
from enum import Enum


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


class SnakeGame:
    def __init__(self, width=10, height=10):
        if width < 10 or height < 10:
            raise ValueError("Width and height must be at least 10.")
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        max_attempts = 100
        for _ in range(max_attempts):
            head_x = random.randint(2, self.width - 3)
            head_y = random.randint(2, self.height - 3)

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
            dx, dy = self.direction.value

            snake_body = [
                (head_x, head_y),
                (head_x - dx, head_y - dy),
                (head_x - 2 * dx, head_y - 2 * dy),
            ]

            # Make sure all segments are in bounds
            if all(0 <= x < self.width and 0 <= y < self.height for x, y in snake_body):
                self.snake = snake_body
                # Initialize grid in memory
                self.grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
                for x, y in self.snake:
                    self.grid[y][x] = 1
                self.spawn_apples()  # spawns both apples into the grid
                self.game_over = False
                return

        # Fallback: center if random fails
        mid_x, mid_y = self.width // 2, self.height // 2
        self.snake = [(mid_x, mid_y), (mid_x - 1, mid_y), (mid_x - 2, mid_y)]
        self.direction = Direction.RIGHT
        self.pending_direction = self.direction
        self.grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
        for x, y in self.snake:
            self.grid[y][x] = 1
        self.spawn_apples()
        self.game_over = False

    def spawn_apples(self):
        self.spawn_green_apple()
        self.spawn_red_apple()

    def spawn_green_apple(self):
        empty_cells = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if self.grid[y][x] == 0 and (x, y) != getattr(self, "red_apple", None)
        ]
        if empty_cells:
            self.green_apple = random.choice(empty_cells)
            self.grid[self.green_apple[1]][self.green_apple[0]] = 2

    def spawn_red_apple(self):
        empty_cells = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if self.grid[y][x] == 0 and (x, y) != getattr(self, "green_apple", None)
        ]
        if empty_cells:
            self.red_apple = random.choice(empty_cells)
            self.grid[self.red_apple[1]][self.red_apple[0]] = 3
        else:
            self.red_apple = None

    def set_direction(self, new_direction: Direction):
        """Change snake direction, ignoring 180-degree reversals."""
        if (
            new_direction.value[0] * -1,
            new_direction.value[1] * -1,
        ) != self.direction.value:
            self.pending_direction = new_direction

    def _pop_snake(self):
        tail = self.snake.pop()
        self.grid[tail[1]][tail[0]] = 0

    def step(self):
        """Advance the game one tick in the current direction."""
        if self.game_over:
            return

        # Apply pending direction
        self.direction = self.pending_direction

        head_x, head_y = self.snake[0]
        dx, dy = self.direction.value
        new_head = (head_x + dx, head_y + dy)

        # Check collisions
        if (new_head in self.snake) or not (
            0 <= new_head[0] < self.width and 0 <= new_head[1] < self.height
        ):
            self.game_over = True
            return

        self.snake.insert(0, new_head)
        self.grid[new_head[1]][new_head[0]] = 1
        if new_head == self.green_apple:
            self.spawn_green_apple()
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
        return

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
