import copy

import pygame
from snake_ai import choose_direction

from ..snake_game.SnakeGame import Direction, SnakeGame

GAME = SnakeGame(10, 10)


def play_game_user(use_box=True):
    pygame.init()
    cell_size = 30
    # Get copy of initial game state
    game = copy.deepcopy(GAME)
    # PYGAME needs a display to capture the keyboard inputs
    screen = (
        pygame.display.set_mode((1, 1))
        if not use_box
        else pygame.display.set_mode((game.width * cell_size, game.height * cell_size))
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
            text = font.render(f"Size: {len(game.snake)}", True, (255, 255, 255))
            screen.blit(text, (game.width * cell_size - text.get_width() - 5, 5))
            pygame.display.flip()

        game.render_terminal()
        clock.tick(5)
    pygame.quit()


def get_direction_ai(snake_view):
    dx, dy = choose_direction(snake_view)
    for d in Direction:
        if d.value == (dx, dy):
            direction = d
            break

    return direction


def play_game_ai():
    game = copy.deepcopy(GAME)
    while not game.game_over:
        snake_view = game.get_snake_view()
        direction = get_direction_ai(snake_view)
        game.set_direction(direction)
        game.step()
        game.render_terminal()


if __name__ == "__main__":
    # play_game_user(True)
    play_game_ai()
