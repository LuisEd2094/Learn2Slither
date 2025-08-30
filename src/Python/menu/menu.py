import sys
from argparse import Namespace

import pygame


class Menu:
    def __init__(self):
        pygame.init()
        self.font = pygame.font.SysFont("Arial", 28)
        self.screen = pygame.display.set_mode((1024, 768))
        self.clock = pygame.time.Clock()
        self.running = True

        # Default values
        self.options = {
            "sessions": 10,
            "learn": False,
            "human_speed": False,
            "pve": False,
            "grid_size": 10,
        }

        self.items = list(self.options.keys())
        self.items.append("START")
        self.selected_index = 0

    def draw_text(self, text, x, y, color=(255, 255, 255)):
        label = self.font.render(text, True, color)
        self.screen.blit(label, (x, y))

    def run(self):
        while self.running:
            self.screen.fill((30, 30, 30))

            # Draw menu items
            for i, item in enumerate(self.items):
                color = (255, 255, 0) if i == self.selected_index else (255, 255, 255)

                if item == "START":
                    text = ">>> START GAME <<<"
                else:
                    value = self.options[item]
                    text = f"{item}: {value}"

                self.draw_text(text, 60, 60 + i * 40, color)

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()

                    if event.key == pygame.K_UP:
                        self.selected_index = (self.selected_index - 1) % len(
                            self.items
                        )
                    elif event.key == pygame.K_DOWN:
                        self.selected_index = (self.selected_index + 1) % len(
                            self.items
                        )

                    elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                        current_item = self.items[self.selected_index]
                        if current_item == "START":
                            self.running = False
                        else:
                            self._modify_option(current_item, toggle_only=True)

                    elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                        current_item = self.items[self.selected_index]
                        if current_item in ["sessions", "grid_size"]:
                            self._modify_numeric(current_item, event.key)

            pygame.display.flip()
            self.clock.tick(30)

        return Namespace(**self.options)

    def _modify_option(self, item, toggle_only=False):
        """Change value depending on type"""
        if isinstance(self.options[item], bool):
            self.options[item] = not self.options[item]

    def _modify_numeric(self, item, key):
        """Adjust integer options with left/right keys"""
        if key == pygame.K_RIGHT:
            self.options[item] += 1
        elif key == pygame.K_LEFT:
            self.options[item] -= 1

        # Clamp values
        if item == "sessions" and self.options[item] < 1:
            self.options[item] = 1
        if item == "grid_size" and self.options[item] < 5:
            self.options[item] = 5
