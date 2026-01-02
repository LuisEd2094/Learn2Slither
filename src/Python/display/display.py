import sys

import pygame

from Python.constants import (
    BACKGROUND_TILE,
    FONT,
    GAME_SPEED,
    LIGHT_BLUE,
    LIGHT_GREEN,
    PVE_OFFSET_X,
    SCREEN_HEIGHT,
    SCREEN_OFFSET_X,
    SCREEN_OFFSET_Y,
    SCREEN_WIDTH,
    SPRITE_SIZE,
    YELLOW_ORANGE,
)
from Python.learn_2_slither import Learn2Slither
from Python.snake_game import Objects, SnakeGame


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
                    pygame.draw.rect(self.screen, (0, 200, 0), rect)
                elif grid[y][x] == Objects.GREEN_APPLE.value:
                    pygame.draw.rect(self.screen, (0, 255, 0), rect)
                elif grid[y][x] == Objects.RED_APPLE.value:
                    pygame.draw.rect(self.screen, (255, 0, 0), rect)
                pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)
        text = f"Size: {len(game.snake)}"
        self.draw_text(text, text_x, 5, YELLOW_ORANGE)

    def render_game(self):
        """Render the current game state(s) to the screen."""
        self.screen.blit(self.background, (0, 0))

        if not self.pve:
            self._update_board(
                self.main_game,
                self.offset_x,
                self.offset_y,
                5,
                self.display_cols,
                self.display_rows,
            )

        else:
            self._update_board(
                self.main_game,
                self.offset_x_left,
                self.offset_y_left,
                5,
                self.display_cols_left,
                self.display_rows_left,
            )
            self._update_board(
                self.secondary_game,
                self.offset_x_right,
                self.offset_y_right,
                self.screen.get_width() // 2 + 5,
                self.display_cols_right,
                self.display_rows_right,
            )

        pygame.display.flip()

    def tick(self):
        """
        Update frame timing when human_speed mode is active.

        Returns:
            The number of game steps that should occur this frame (0 or more).
            When human_speed is False, returns None immediately.
        """
        if not self.human_speed:
            return
        dt = self.clock.tick(60) / 1000.0
        self._time_accumulator += dt

        steps = 0
        step_interval = 1.0 / self.clock_tick

        while self._time_accumulator >= step_interval:
            self._time_accumulator -= step_interval
            steps += 1

        return steps

    def quit(self):
        """Clean up pygame resources and exit the program."""
        pygame.quit()
        sys.exit()
