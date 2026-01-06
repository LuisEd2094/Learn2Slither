import sys

import pygame

from Python.constants import (
    BACKGROUND_TILE,
    BAD_FOOD_SPRITE_PATH,
    DRAW_GRID,
    FONT,
    FOOD_SPRITE_PATH,
    GAME_SPEED,
    GROUND_SNOW_MIDDLE,
    LIGHT_BLUE,
    LIGHT_GREEN,
    PVE_OFFSET_X,
    SCREEN_HEIGHT,
    SCREEN_OFFSET_X,
    SCREEN_OFFSET_Y,
    SCREEN_WIDTH,
    SNAKE_BODY_CORNER_02,
    SNAKE_BODY_CORNER_03,
    SNAKE_BODY_HORIZONTAL_00,
    SNAKE_BODY_HORIZONTAL_01,
    SNAKE_BODY_LEFT_CONNECTOR,
    SNAKE_BODY_RIGHT_CONNECTOR,
    SNAKE_BODY_VERTICAL_00,
    SNAKE_BODY_VERTICAL_01,
    SNAKE_HEAD_00,
    SNAKE_HEAD_01,
    SNAKE_TAIL_00,
    SNAKE_TAIL_01,
    SNAKE_TAIL_02,
    SNAKE_TAIL_03,
    SNAKE_TAIL_04,
    SPRITE_SIZE,
    WALL_BOT_END,
    WALL_CENTER_HORI,
    WALL_CENTER_VER,
    WALL_CORNER,
    WALL_LEFT_END,
    WALL_RIGHT_END,
    WALL_TOP_END,
    YELLOW_ORANGE,
)
from Python.learn_2_slither import Learn2Slither
from Python.snake_game import Direction, Objects, SnakeGame


class Display:
    _instance = None
    _SPRITE_SIZE = SPRITE_SIZE

    def __init__(self):
        """
        Initialize the Display singleton with pygame setup and layout calculations.

        Raises:
            RuntimeError: If Display is instantiated directly instead of using get_instance().
        """
        if Display._instance is not None:
            raise RuntimeError(
                "Use Display.get_instance() instead of instantiating directly"
            )
        pygame.init()
        self.font = pygame.font.Font(
            FONT,
            28,
        )
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True
        self.human_speed = True
        self.clock_tick = GAME_SPEED
        self._time_accumulator = 0
        self.background = self._get_background(BACKGROUND_TILE)
        self.food_sprite = self._load_sprite(FOOD_SPRITE_PATH)
        self.bad_food_sprite = self._load_sprite(BAD_FOOD_SPRITE_PATH)
        self._load_body_sprites()
        self._load_wall_sprites()
        self._load_ground_tiles()
        self._animation_frame = 0
        self._calculate_layout_dimensions()

    def _get_background(self, file):
        """
        Load and tile a background image to fit the entire screen.

        Args:
            file: Path to the background tile image.

        Returns:
            A pygame.Surface containing the tiled background.
        """
        background_tile = pygame.image.load(file)
        screen_w, screen_h = self.screen.get_size()
        background = pygame.Surface((screen_w, screen_h))
        background.blit(background_tile, (0, 0))
        return background

    def _load_sprite(self, file):
        """
        Load and scale a sprite image to match SPRITE_SIZE.

        Args:
            file: Path to the sprite image.

        Returns:
            A pygame.Surface containing the scaled sprite.
        """
        sprite = pygame.image.load(file)
        return pygame.transform.scale(sprite, (self._SPRITE_SIZE, self._SPRITE_SIZE))

    def _load_body_sprites(self):
        """Load all snake body sprites for horizontal and vertical animation."""
        self.body_horizontal = [
            self._load_sprite(SNAKE_BODY_HORIZONTAL_00),
            self._load_sprite(SNAKE_BODY_HORIZONTAL_01),
        ]
        self.body_vertical = [
            self._load_sprite(SNAKE_BODY_VERTICAL_00),
            self._load_sprite(SNAKE_BODY_VERTICAL_01),
        ]
        self.corner_connectors = [
            self._load_sprite(SNAKE_BODY_LEFT_CONNECTOR),
            self._load_sprite(SNAKE_BODY_RIGHT_CONNECTOR),
            self._load_sprite(SNAKE_BODY_CORNER_02),
            self._load_sprite(SNAKE_BODY_CORNER_03),
        ]

        base_head_sprites = [
            self._load_sprite(SNAKE_HEAD_00),
            self._load_sprite(SNAKE_HEAD_01),
        ]

        self.head_sprites = {
            Direction.DOWN.value: base_head_sprites,
            Direction.UP.value: [
                pygame.transform.rotate(sprite, 180) for sprite in base_head_sprites
            ],
            Direction.LEFT.value: [
                pygame.transform.rotate(sprite, -90) for sprite in base_head_sprites
            ],
            Direction.RIGHT.value: [
                pygame.transform.rotate(sprite, 90) for sprite in base_head_sprites
            ],
        }

        base_tail_sprites = [
            self._load_sprite(SNAKE_TAIL_00),
            self._load_sprite(SNAKE_TAIL_01),
            self._load_sprite(SNAKE_TAIL_02),
            self._load_sprite(SNAKE_TAIL_03),
            self._load_sprite(SNAKE_TAIL_04),
        ]

        self.tail_sprites = {
            Direction.DOWN.value: base_tail_sprites,
            Direction.UP.value: [
                pygame.transform.rotate(sprite, 180) for sprite in base_tail_sprites
            ],
            Direction.LEFT.value: [
                pygame.transform.rotate(sprite, -90) for sprite in base_tail_sprites
            ],
            Direction.RIGHT.value: [
                pygame.transform.rotate(sprite, 90) for sprite in base_tail_sprites
            ],
        }

    def _load_wall_sprites(self):
        """Load all wall sprites for border rendering."""
        self.wall_corner = self._load_sprite(WALL_CORNER)
        self.wall_center_hori = self._load_sprite(WALL_CENTER_HORI)
        self.wall_center_ver = self._load_sprite(WALL_CENTER_VER)
        self.wall_left_end = self._load_sprite(WALL_LEFT_END)
        self.wall_right_end = self._load_sprite(WALL_RIGHT_END)
        self.wall_top_end = self._load_sprite(WALL_TOP_END)
        self.wall_bot_end = self._load_sprite(WALL_BOT_END)

    def _load_ground_tiles(self):
        """Load ground tile sprites for grid interior."""
        self.ground_tiles = [
            self._load_sprite(GROUND_SNOW_MIDDLE),
        ]

    def _render_ground_tiles(
        self, offset_x, offset_y, display_cols, display_rows, grid_id=0
    ):
        """
        Render ground tiles inside the playable grid.

        Args:
            offset_x: Horizontal pixel offset for the board.
            offset_y: Vertical pixel offset for the board.
            display_cols: Number of columns in the playable grid.
            display_rows: Number of rows in the playable grid.
            grid_id: Identifier for the grid (0 for main, 1 for secondary in PVE).
        """
        tile = self.ground_tiles[0]
        for y in range(display_rows):
            for x in range(display_cols):
                self.screen.blit(
                    tile,
                    (
                        offset_x + x * self._SPRITE_SIZE,
                        offset_y + y * self._SPRITE_SIZE,
                    ),
                )

    def _render_wall_corners(self, offset_x, offset_y, display_cols, display_rows):
        """
        Render corner sprites at all four corners of the grid.

        Args:
            offset_x: Horizontal pixel offset for the board.
            offset_y: Vertical pixel offset for the board.
            display_cols: Number of columns in the playable grid.
            display_rows: Number of rows in the playable grid.
        """
        self.screen.blit(
            self.wall_corner,
            (offset_x - self._SPRITE_SIZE, offset_y - self._SPRITE_SIZE),
        )
        self.screen.blit(
            self.wall_corner,
            (offset_x + display_cols * self._SPRITE_SIZE, offset_y - self._SPRITE_SIZE),
        )
        self.screen.blit(
            self.wall_corner,
            (
                offset_x + display_cols * self._SPRITE_SIZE,
                offset_y + display_rows * self._SPRITE_SIZE,
            ),
        )
        self.screen.blit(
            self.wall_corner,
            (offset_x - self._SPRITE_SIZE, offset_y + display_rows * self._SPRITE_SIZE),
        )

    def _render_top_wall(self, offset_x, offset_y, display_cols):
        """
        Render top wall with left_end, center_hori tiles, and right_end.

        Args:
            offset_x: Horizontal pixel offset for the board.
            offset_y: Vertical pixel offset for the board.
            display_cols: Number of columns in the playable grid.
        """
        self.screen.blit(self.wall_left_end, (offset_x, offset_y - self._SPRITE_SIZE))
        for i in range(1, display_cols - 1):
            self.screen.blit(
                self.wall_center_hori,
                (offset_x + i * self._SPRITE_SIZE, offset_y - self._SPRITE_SIZE),
            )
        self.screen.blit(
            self.wall_right_end,
            (
                offset_x + (display_cols - 1) * self._SPRITE_SIZE,
                offset_y - self._SPRITE_SIZE,
            ),
        )

    def _render_bottom_wall(self, offset_x, offset_y, display_cols, display_rows):
        """
        Render bottom wall with right_end, center_hori tiles, and left_end.

        Args:
            offset_x: Horizontal pixel offset for the board.
            offset_y: Vertical pixel offset for the board.
            display_cols: Number of columns in the playable grid.
            display_rows: Number of rows in the playable grid.
        """
        self.screen.blit(
            self.wall_left_end, (offset_x, offset_y + display_rows * self._SPRITE_SIZE)
        )
        for i in range(1, display_cols - 1):
            self.screen.blit(
                self.wall_center_hori,
                (
                    offset_x + i * self._SPRITE_SIZE,
                    offset_y + display_rows * self._SPRITE_SIZE,
                ),
            )
        self.screen.blit(
            self.wall_right_end,
            (
                offset_x + (display_cols - 1) * self._SPRITE_SIZE,
                offset_y + display_rows * self._SPRITE_SIZE,
            ),
        )

    def _render_left_wall(self, offset_x, offset_y, display_rows):
        """
        Render left wall with top_end, center_ver tiles, and bot_end.

        Args:
            offset_x: Horizontal pixel offset for the board.
            offset_y: Vertical pixel offset for the board.
            display_rows: Number of rows in the playable grid.
        """
        self.screen.blit(self.wall_top_end, (offset_x - self._SPRITE_SIZE, offset_y))
        for i in range(1, display_rows - 1):
            self.screen.blit(
                self.wall_center_ver,
                (offset_x - self._SPRITE_SIZE, offset_y + i * self._SPRITE_SIZE),
            )
        self.screen.blit(
            self.wall_bot_end,
            (
                offset_x - self._SPRITE_SIZE,
                offset_y + (display_rows - 1) * self._SPRITE_SIZE,
            ),
        )

    def _render_right_wall(self, offset_x, offset_y, display_cols, display_rows):
        """
        Render right wall with bot_end, center_ver tiles, and top_end.

        Args:
            offset_x: Horizontal pixel offset for the board.
            offset_y: Vertical pixel offset for the board.
            display_cols: Number of columns in the playable grid.
            display_rows: Number of rows in the playable grid.
        """
        self.screen.blit(
            self.wall_top_end, (offset_x + display_cols * self._SPRITE_SIZE, offset_y)
        )
        for i in range(1, display_rows - 1):
            self.screen.blit(
                self.wall_center_ver,
                (
                    offset_x + display_cols * self._SPRITE_SIZE,
                    offset_y + i * self._SPRITE_SIZE,
                ),
            )
        self.screen.blit(
            self.wall_bot_end,
            (
                offset_x + display_cols * self._SPRITE_SIZE,
                offset_y + (display_rows - 1) * self._SPRITE_SIZE,
            ),
        )

    def _render_walls(self, offset_x, offset_y, display_cols, display_rows):
        """
        Render complete wall border around the playable grid.

        Walls are placed outside the grid boundaries with corners, end pieces, and center tiles.

        Args:
            offset_x: Horizontal pixel offset for the board.
            offset_y: Vertical pixel offset for the board.
            display_cols: Number of columns in the playable grid.
            display_rows: Number of rows in the playable grid.
        """
        self._render_wall_corners(offset_x, offset_y, display_cols, display_rows)
        self._render_top_wall(offset_x, offset_y, display_cols)
        self._render_bottom_wall(offset_x, offset_y, display_cols, display_rows)
        self._render_left_wall(offset_x, offset_y, display_rows)
        self._render_right_wall(offset_x, offset_y, display_cols, display_rows)

    def _get_body_sprite(self, is_horizontal, segment_index):
        """
        Get the current frame of the snake body sprite based on animation state.

        Args:
            is_horizontal: Boolean indicating if the body segment is horizontal.
            segment_index: Index of the snake segment for alternating pattern.

        Returns:
            A pygame.Surface containing the current animation frame sprite.
        """
        frame = (self._animation_frame + segment_index) % 2
        if is_horizontal:
            return self.body_horizontal[frame]
        else:
            return self.body_vertical[frame]

    def _get_head_sprite(self, game):
        """
        Get the rotated head sprite based on snake direction.

        Args:
            game: The SnakeGame instance containing direction information.

        Returns:
            A pygame.Surface containing the rotated head sprite.
        """
        frame = self._animation_frame % 2
        direction = game.get_heading()
        return self.head_sprites[direction][frame]

    def _get_tail_direction(self, snake):
        """
        Determine the direction the tail is pointing based on last two segments.

        Tail points backward (opposite of movement direction).

        Args:
            snake: List of (x, y) tuples representing snake body positions (must have at least 2 segments).

        Returns:
            Direction value tuple for tail orientation.
        """
        tail_pos = snake[-1]
        before_tail_pos = snake[-2]

        dx = tail_pos[0] - before_tail_pos[0]
        dy = tail_pos[1] - before_tail_pos[1]

        if dy > 0:
            return Direction.UP.value
        elif dy < 0:
            return Direction.DOWN.value
        elif dx > 0:
            return Direction.LEFT.value
        elif dx < 0:
            return Direction.RIGHT.value

        return Direction.UP.value

    def _get_tail_sprite(self, game):
        """
        Get the rotated tail sprite with animation cycling through 5 frames.

        Args:
            game: The SnakeGame instance containing snake positions.

        Returns:
            A pygame.Surface containing the rotated tail sprite, or None if snake is too short.
        """
        if len(game.snake) < 2:
            return None

        frame = self._animation_frame % 5
        direction = self._get_tail_direction(game.snake)
        return self.tail_sprites[direction][frame]

    def _update_animation_frame(self):
        """Update the animation frame counter for body sprite cycling."""
        self._animation_frame += 1

    def _calculate_layout_dimensions(self):
        """
        Calculate the maximum board dimensions based on 32px sprites.
        Accounts for SCREEN_OFFSET_Y at the top for score display.
        """
        available_width = SCREEN_WIDTH - SCREEN_OFFSET_X
        available_height = SCREEN_HEIGHT - SCREEN_OFFSET_Y

        self.max_cols_single = available_width // self._SPRITE_SIZE
        self.max_rows_single = available_height // self._SPRITE_SIZE

        width_for_two_grids = available_width - PVE_OFFSET_X
        width_per_grid = width_for_two_grids // 2

        self.max_cols_dual = width_per_grid // self._SPRITE_SIZE
        self.max_rows_dual = available_height // self._SPRITE_SIZE

    def _init_single_game_layout(self):
        """
        Initialize grid layout for single-player mode.

        Centers the game board horizontally and vertically within the display,
        accounting for offset gaps.
        """
        self.display_cols = self.main_game.width
        self.display_rows = self.main_game.height

        grid_pixel_width = self.display_cols * self._SPRITE_SIZE
        grid_pixel_height = self.display_rows * self._SPRITE_SIZE

        left_offset = SCREEN_OFFSET_X // 2
        available_width = SCREEN_WIDTH - SCREEN_OFFSET_X

        top_offset = SCREEN_OFFSET_Y // 2
        available_height = SCREEN_HEIGHT - SCREEN_OFFSET_Y

        self.offset_x = left_offset + (available_width - grid_pixel_width) // 2
        self.offset_y = top_offset + (available_height - grid_pixel_height) // 2

    def _init_pve_layout(self):
        """
        Initialize grid layout for player vs AI mode (side-by-side).

        Positions main and secondary game boards side-by-side with proper spacing.
        Accounts for offset gaps on all sides.
        """
        self.display_cols_left = self.main_game.width
        self.display_rows_left = self.main_game.height
        self.display_cols_right = self.secondary_game.width
        self.display_rows_right = self.secondary_game.height

        grid_pixel_height_main = self.display_rows_left * self._SPRITE_SIZE
        grid_pixel_height_secondary = self.display_rows_right * self._SPRITE_SIZE

        left_offset = SCREEN_OFFSET_X // 2
        available_width = SCREEN_WIDTH - SCREEN_OFFSET_X

        top_offset = SCREEN_OFFSET_Y // 2
        available_height = SCREEN_HEIGHT - SCREEN_OFFSET_Y

        grid_pixel_width_main = self.display_cols_left * self._SPRITE_SIZE
        half_width_available = (available_width - PVE_OFFSET_X) // 2

        self.offset_x_left = (
            left_offset + (half_width_available - grid_pixel_width_main) // 2
        )
        self.offset_y_left = (
            top_offset + (available_height - grid_pixel_height_main) // 2
        )

        grid_pixel_width_secondary = self.display_cols_right * self._SPRITE_SIZE
        self.offset_x_right = (
            left_offset
            + half_width_available
            + PVE_OFFSET_X // 2
            + (half_width_available - grid_pixel_width_secondary) // 2
        )
        self.offset_y_right = (
            top_offset + (available_height - grid_pixel_height_secondary) // 2
        )

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
        text_surf = self.font.render(text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(topleft=(x, y))

        gradient = pygame.Surface(text_surf.get_size(), pygame.SRCALPHA)
        height = text_surf.get_height()

        for y_pos in range(height):
            total_segments = len(colors) - 1
            segment_height = height / total_segments
            segment_index = int(y_pos // segment_height)
            if segment_index >= total_segments:
                segment_index = total_segments - 1

            c1 = colors[segment_index]
            c2 = colors[segment_index + 1]

            factor = (y_pos - segment_index * segment_height) / segment_height
            r = int(c1[0] + (c2[0] - c1[0]) * factor)
            g = int(c1[1] + (c2[1] - c1[1]) * factor)
            b = int(c1[2] + (c2[2] - c1[2]) * factor)

            pygame.draw.line(
                gradient, (r, g, b), (0, y_pos), (text_surf.get_width(), y_pos)
            )

        gradient.blit(text_surf, (0, 0), None, pygame.BLEND_RGBA_MULT)

        self.screen.blit(gradient, text_rect)

    def fill(self, color=(0, 0, 0)):
        """
        Fill the entire screen with a solid color.

        Args:
            color: RGB tuple for the fill color. Defaults to black.
        """
        self.screen.fill(color)

    def flip(self):
        """Update the display to show all drawn elements."""
        pygame.display.flip()

    def display_menu(self, selected_index, items, options, difficulty_levels):
        """
        Render the main menu with options and highlight the selected item.

        Args:
            selected_index: The index of the currently selected menu item.
            items: List of menu item keys (e.g., ['pve', 'visuals', 'START']).
            options: Dictionary mapping item keys to their current values.
            difficulty_levels: Dictionary mapping difficulty codes to display names.
        """
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
            self._write_menu_option(text, i, color)

    def _write_menu_option(self, text, i, color):
        """
        Render a single menu option text at the specified row.

        Args:
            text: The menu option text to display.
            i: The row index (0-based) of the menu option.
            color: RGB tuple for the text color.
        """
        self.draw_text(text, 60, 60 + i * 40, color)

    def init_game(self, l2s: Learn2Slither):
        """
        Initialize display settings based on the Learn2Slither game configuration.

        Args:
            l2s: The Learn2Slither instance containing game settings and board references.

        Raises:
            ValueError: If game board dimensions exceed display constraints.
        """
        self.human_speed = l2s.human_speed
        self.visuals = l2s.visuals
        self.main_game = l2s.main_game
        self.secondary_game = l2s.secondary_game
        self.pve = l2s.pve
        self.l2s = l2s

        self._validate_game_dimensions()

        if not self.pve:
            self._init_single_game_layout()
        else:
            self._init_pve_layout()

    def _validate_game_dimensions(self):
        """Validate that game board(s) fit within display constraints."""
        if not self.pve:
            # Single game mode
            if self.main_game.width > self.max_cols_single:
                raise ValueError(
                    f"Game width ({self.main_game.width}) exceeds maximum displayable width "
                    f"({self.max_cols_single}). Reduce board width or increase SCREEN_WIDTH."
                )
            if self.main_game.height > self.max_rows_single:
                raise ValueError(
                    f"Game height ({self.main_game.height}) exceeds maximum displayable height "
                    f"({self.max_rows_single}). Reduce board height or increase SCREEN_HEIGHT."
                )
        else:
            # Dual game mode (PVE)
            if self.main_game.width > self.max_cols_dual:
                raise ValueError(
                    f"Main game width ({self.main_game.width}) exceeds maximum displayable width "
                    f"({self.max_cols_dual}) for dual mode. Reduce board width or adjust screen settings."
                )
            if self.main_game.height > self.max_rows_dual:
                raise ValueError(
                    f"Main game height ({self.main_game.height}) exceeds maximum displayable height "
                    f"({self.max_rows_dual}). Reduce board height or increase SCREEN_HEIGHT."
                )
            if self.secondary_game.width > self.max_cols_dual:
                raise ValueError(
                    f"Secondary game width ({self.secondary_game.width}) exceeds maximum displayable width "
                    f"({self.max_cols_dual}) for dual mode. Reduce board width or adjust screen settings."
                )
            if self.secondary_game.height > self.max_rows_dual:
                raise ValueError(
                    f"Secondary game height ({self.secondary_game.height}) exceeds maximum displayable height "
                    f"({self.max_rows_dual}). Reduce board height or increase SCREEN_HEIGHT."
                )

    def _is_segment_horizontal(self, snake, segment_index):
        """
        Determine if a snake segment is moving horizontally.

        Args:
            snake: List of (x, y) tuples representing snake body positions.
            segment_index: Index of the segment to check.

        Returns:
            True if the segment is moving horizontally, False if vertical.
        """
        if segment_index == 0:
            if len(snake) > 1:
                return snake[0][0] != snake[1][0]
            return True
        return snake[segment_index][0] != snake[segment_index - 1][0]

    def _get_snake_segment_index(self, snake, x, y):
        """
        Find the index of a snake segment at the given grid position.

        Args:
            snake: List of (x, y) tuples representing snake body positions.
            x: Grid x coordinate.
            y: Grid y coordinate.

        Returns:
            The index of the segment, or 0 if not found.
        """
        for i, (sx, sy) in enumerate(snake):
            if sx == x and sy == y:
                return i
        return 0

    def _is_corner_segment(self, snake, segment_index):
        """
        Determine if a snake segment is at a corner (turn point).

        A corner exists when the direction changes between consecutive segments.

        Args:
            snake: List of (x, y) tuples representing snake body positions.
            segment_index: Index of the segment to check.

        Returns:
            True if the segment is at a corner, False otherwise.
        """
        if segment_index == 0 or segment_index >= len(snake) - 1:
            return False

        prev_x, prev_y = snake[segment_index - 1]
        curr_x, curr_y = snake[segment_index]
        next_x, next_y = snake[segment_index + 1]

        coming_from_horizontal = prev_x != curr_x
        going_to_horizontal = curr_x != next_x

        return coming_from_horizontal != going_to_horizontal

    def _get_corner_direction(self, snake, segment_index):
        """
        Determine the correct connector sprite for a corner segment.

        Maps direction combinations to 4 pre-rotated connector sprites:
        00: down->up-right OR right->down
        01: top->right OR left->up
        02: left->top OR top->left
        03: bottom->left OR left->down

        Args:
            snake: List of (x, y) tuples representing snake body positions.
            segment_index: Index of the corner segment.

        Returns:
            Connector index (0-3) for the correct pre-rotated sprite.
        """
        prev_x, prev_y = snake[segment_index - 1]
        curr_x, curr_y = snake[segment_index]
        next_x, next_y = snake[segment_index + 1]

        incoming_dx = curr_x - prev_x
        incoming_dy = curr_y - prev_y
        outgoing_dx = next_x - curr_x
        outgoing_dy = next_y - curr_y

        if (incoming_dy < 0 and outgoing_dx > 0) or (
            incoming_dx > 0 and outgoing_dy > 0
        ):
            return 0
        elif (incoming_dy > 0 and outgoing_dx > 0) or (
            incoming_dx < 0 and outgoing_dy < 0
        ):
            return 1
        elif (incoming_dx < 0 and outgoing_dy < 0) or (
            incoming_dy > 0 and outgoing_dx < 0
        ):
            return 2
        elif (incoming_dy < 0 and outgoing_dx < 0) or (
            incoming_dx < 0 and outgoing_dy > 0
        ):
            return 3

        return 0

    def _render_snake_segment(self, game, x, y, segment_index, offset_x, offset_y):
        """
        Render a single snake body segment with appropriate animation.

        Args:
            game: The SnakeGame instance.
            x: Grid x coordinate.
            y: Grid y coordinate.
            segment_index: Index of the snake segment.
            offset_x: Pixel offset for x.
            offset_y: Pixel offset for y.
        """
        rect = pygame.Rect(
            offset_x + x * self._SPRITE_SIZE,
            offset_y + y * self._SPRITE_SIZE,
            self._SPRITE_SIZE,
            self._SPRITE_SIZE,
        )

        if segment_index == 0:
            sprite = self._get_head_sprite(game)
        elif segment_index == len(game.snake) - 1:
            sprite = self._get_tail_sprite(game)
        elif self._is_corner_segment(game.snake, segment_index):
            corner_type = self._get_corner_direction(game.snake, segment_index)
            sprite = self.corner_connectors[corner_type]
        else:
            is_horizontal = self._is_segment_horizontal(game.snake, segment_index)
            sprite = self._get_body_sprite(is_horizontal, segment_index)

        if sprite is not None:
            self.screen.blit(sprite, rect)

    def _update_board(
        self, game: SnakeGame, offset_x, offset_y, text_x, display_cols, display_rows
    ):
        """
        Render a single game board with snake, apples, and grid lines.

        Args:
            game: The SnakeGame instance to render.
            offset_x: Horizontal pixel offset for the board.
            offset_y: Vertical pixel offset for the board.
            text_x: Horizontal pixel position for the score text.
            display_cols: Number of columns to display.
            display_rows: Number of rows to display.
        """
        grid = game.get_state()
        for y in range(display_rows):
            for x in range(display_cols):
                rect = pygame.Rect(
                    offset_x + x * self._SPRITE_SIZE,
                    offset_y + y * self._SPRITE_SIZE,
                    self._SPRITE_SIZE,
                    self._SPRITE_SIZE,
                )
                if grid[y][x] == Objects.SNAKE.value:
                    segment_index = self._get_snake_segment_index(game.snake, x, y)
                    self._render_snake_segment(
                        game, x, y, segment_index, offset_x, offset_y
                    )
                elif grid[y][x] == Objects.GREEN_APPLE.value:
                    self.screen.blit(self.food_sprite, rect)
                elif grid[y][x] == Objects.RED_APPLE.value:
                    self.screen.blit(self.bad_food_sprite, rect)
                if DRAW_GRID:
                    pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)
        text = f"Size: {len(game.snake)}"
        self.draw_text(text, text_x, 5, YELLOW_ORANGE)

    def render_game(self):
        """Render the current game state(s) to the screen."""
        self.screen.blit(self.background, (0, 0))

        if not self.pve:
            self._render_ground_tiles(
                self.offset_x,
                self.offset_y,
                self.display_cols,
                self.display_rows,
                grid_id=0,
            )
            self._render_walls(
                self.offset_x,
                self.offset_y,
                self.display_cols,
                self.display_rows,
            )
            self._update_board(
                self.main_game,
                self.offset_x,
                self.offset_y,
                5,
                self.display_cols,
                self.display_rows,
            )

        else:
            self._render_ground_tiles(
                self.offset_x_left,
                self.offset_y_left,
                self.display_cols_left,
                self.display_rows_left,
                grid_id=0,
            )
            self._render_walls(
                self.offset_x_left,
                self.offset_y_left,
                self.display_cols_left,
                self.display_rows_left,
            )
            self._update_board(
                self.main_game,
                self.offset_x_left,
                self.offset_y_left,
                5,
                self.display_cols_left,
                self.display_rows_left,
            )
            self._render_ground_tiles(
                self.offset_x_right,
                self.offset_y_right,
                self.display_cols_right,
                self.display_rows_right,
                grid_id=1,
            )
            self._render_walls(
                self.offset_x_right,
                self.offset_y_right,
                self.display_cols_right,
                self.display_rows_right,
            )
            self._update_board(
                self.secondary_game,
                self.offset_x_right,
                self.offset_y_right,
                self.screen.get_width() // 2 + 5,
                self.display_cols_right,
                self.display_rows_right,
            )

        self._update_animation_frame()
        pygame.display.flip()

    def tick(self):
        """
        Update frame timing when human_speed mode is active.

        Throttles game speed to make it visible for humans.
        When human_speed is False, returns 1 step without throttling.

        Returns:
            int: Number of steps to process (always 1).
        """
        if not self.human_speed:
            return 1
        self.clock.tick(self.clock_tick)
        return 1

    def quit(self):
        """Clean up pygame resources and exit the program."""
        pygame.quit()
        sys.exit()
