from argparse import Namespace

import pygame

from Python.display import Display
from Python.learn_2_slither import Learn2Slither


class Menu:
    DIFFICULTY_LEVELS = {1: "easy", 2: "normal", 3: "hard"}

    def __init__(self):
        # Display also inits pygame
        self.display = Display.get_instance()

        # Default values
        self.options = {
            "sessions": 10,
            "learn": False,
            "human_speed": False,
            "pve": False,
            "grid_size": 10,
            "difficulty": 1,
        }

        self.items = list(self.options.keys())
        self.items.append("START")
        self.selected_index = 0
        self.running = True

    def run(self):
        while self.running:
            self.display.fill()
            # Draw menu items
            for i, item in enumerate(self.items):
                color = (255, 255, 0) if i == self.selected_index else (255, 255, 255)

                if item == "START":
                    text = ">>> START GAME <<<"
                else:
                    value = self.options[item]
                    if item == "difficulty":
                        value = self.DIFFICULTY_LEVELS.get(value, "unknown")
                    text = f"{item}: {value}"
                self.display.display_menu(text, i, color)

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.display.quit()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        self.display.quit()

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
                        if current_item in ["sessions", "grid_size", "difficulty"]:
                            self._modify_numeric(current_item, event.key)
            self.display.flip()
            self.display.tick()
        self.options["difficulty"] = self.DIFFICULTY_LEVELS.get(
            self.options["difficulty"], "unknown"
        )

        l2s = Learn2Slither(Namespace(**self.options))
        l2s.run()

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
        if item == "grid_size" and self.options[item] < 10:
            self.options[item] = 10
        if item == "difficulty":
            self.options[item] = max(1, min(3, self.options[item]))
