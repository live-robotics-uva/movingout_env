# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "gymnasium",
#   "cffi",
#   "pymunk",
#   "yaml"
# ]
# ///

import asyncio
import pygame
import pygame.surfarray
import numpy
# --- Constants and Configuration ---
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
# We will import AVAILABLE_MAPS later, inside the main function.

# --- UI Component Classes ---

class Button:
    """A simple, reusable button class for Pygame UI."""
    def __init__(self, x, y, width, height, text, font, base_color=(70, 70, 120), hover_color=(100, 100, 150)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = str(text)
        self.font = font
        self.base_color = base_color
        self.hover_color = hover_color
        self.current_color = base_color

    def draw(self, screen):
        pygame.draw.rect(screen, self.current_color, self.rect, border_radius=8)
        text_surf = self.font.render(self.text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def update(self, mouse_pos):
        if self.rect.collidepoint(mouse_pos):
            self.current_color = self.hover_color
        else:
            self.current_color = self.base_color

    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(event.pos)
        return False

class OptionSelector:
    """A group of buttons where only one can be selected at a time."""
    def __init__(self, x, y, options, font, button_height=40, button_margin=10):
        self.buttons = []
        self.selected_option = options[0]
        self.selected_color = (80, 180, 80)  # Green for selected
        self.base_color = (70, 70, 120)
        self.hover_color = (100, 100, 150)
        
        current_y = y
        for option in options:
            button = Button(x, current_y, 250, button_height, option, font, self.base_color, self.hover_color)
            self.buttons.append(button)
            current_y += button_height + button_margin

    def handle_events(self, event):
        for button in self.buttons:
            if button.is_clicked(event):
                self.selected_option = button.text
                print(f"Selected: {self.selected_option}")

    def update(self, mouse_pos):
        for button in self.buttons:
            button.update(mouse_pos)
            if button.text == self.selected_option:
                # Override color if selected, but still allow hover effect
                if not button.rect.collidepoint(mouse_pos):
                    button.current_color = self.selected_color
            else:
                 if not button.rect.collidepoint(mouse_pos):
                    button.current_color = self.base_color


    def draw(self, screen):
        for button in self.buttons:
            button.draw(screen)

class TextInputBox:
    """A box for user text input, filtered for numbers."""
    def __init__(self, x, y, width, height, font, default_text="10"):
        self.rect = pygame.Rect(x, y, width, height)
        self.font = font
        self.text = default_text
        self.active = False
        self.active_color = (200, 200, 255)
        self.inactive_color = (150, 150, 150)
        self.current_color = self.inactive_color

    def handle_events(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
            self.current_color = self.active_color if self.active else self.inactive_color
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.unicode.isdigit():
                self.text += event.unicode

    def draw(self, screen):
        pygame.draw.rect(screen, (30, 30, 30), self.rect)
        pygame.draw.rect(screen, self.current_color, self.rect, 2, border_radius=5)
        text_surf = self.font.render(self.text, True, (255, 255, 255))
        screen.blit(text_surf, (self.rect.x + 10, self.rect.y + 5))

# --- Main Application Class ---

class App:
    """Manages the application state (Config vs Game) and main loop."""
    def __init__(self, screen):
        self.screen = screen
        self.game_state = "CONFIG"  # Can be "CONFIG" or "GAME"
        self.clock = pygame.time.Clock()
        self.config = {
            "map": None,
            "controller": "Keyboard",
            "fps": 1000
        }
        self.init_config_ui()

    def init_config_ui(self):
        """Initializes all UI elements for the configuration screen."""
        self.font_title = pygame.font.Font(None, 48)
        self.font_label = pygame.font.Font(None, 36)
        self.font_option = pygame.font.Font(None, 32)

        # Map Selector
        from moving_out.env_parameters import AVAILABLE_MAPS
        self.map_options = list(AVAILABLE_MAPS.keys())
        self.config["map"] = self.map_options[0]
        self.map_selector = OptionSelector(50, 100, self.map_options, self.font_option)

        # Controller Selector
        self.controller_options = ["Keyboard", "Joystick"]
        self.controller_selector = OptionSelector(320, 100, self.controller_options, self.font_option)

        # FPS Input
        self.fps_input = TextInputBox(320, 350, 150, 40, self.font_option)
        
        # Start Button
        self.start_button = Button(200, 500, 200, 50, "Start Game", self.font_label, (50, 150, 50), (80, 180, 80))

    def run_config_state(self, events):
        """Handles logic for the configuration screen."""
        mouse_pos = pygame.mouse.get_pos()
        
        for event in events:
            self.map_selector.handle_events(event)
            self.controller_selector.handle_events(event)
            self.fps_input.handle_events(event)
            if self.start_button.is_clicked(event):
                # Save settings and switch to game state
                self.config["map"] = self.map_selector.selected_option
                self.config["controller"] = self.controller_selector.selected_option
                try:
                    self.config["fps"] = int(self.fps_input.text)
                except ValueError:
                    self.config["fps"] = 10 # Default on error
                print(f"Starting game with config: {self.config}")
                self.game_state = "GAME"
                self.init_game_state() # Initialize the game
                return

        # Update UI elements
        self.map_selector.update(mouse_pos)
        self.controller_selector.update(mouse_pos)
        self.start_button.update(mouse_pos)

        # Draw UI
        self.screen.fill((20, 20, 70))
        title_surf = self.font_title.render("Game Configuration", True, (255, 255, 255))
        self.screen.blit(title_surf, (SCREEN_WIDTH / 2 - title_surf.get_width() / 2, 20))
        
        map_label = self.font_label.render("Map", True, (255, 255, 255))
        self.screen.blit(map_label, (50, 60))
        self.map_selector.draw(self.screen)

        controller_label = self.font_label.render("Controller", True, (255, 255, 255))
        self.screen.blit(controller_label, (320, 60))
        self.controller_selector.draw(self.screen)

        fps_label = self.font_label.render("FPS", True, (255, 255, 255))
        self.screen.blit(fps_label, (320, 310))
        self.fps_input.draw(self.screen)
        
        self.start_button.draw(self.screen)

    def init_game_state(self):
        """Initializes the environment and controller based on config."""
        from moving_out.benchmarks.moving_out import MovingOutEnv
        from moving_out.utils.keyboard_controller import KeyboardController
        from moving_out.utils.joystick_controller import JoystickController # Uncomment when ready

        print(f"Using map '{self.config['map']}' to create environment...")
        self.env = MovingOutEnv(map_name=self.config["map"], reward_setting="sparse")
        self.env.reset()

        if self.config["controller"] == "Keyboard":
            self.controller = KeyboardController()
        else:
            self.controller = JoystickController() # Uncomment when ready
            
        # UI for the game screen
        self.reset_button = Button(10, 10, 100, 40, "Reset", self.font_option)
        self.config_button = Button(120, 10, 150, 40, "Back to Config", self.font_option)

    def run_game_state(self, events):
        """Handles logic for the main game screen."""
        mouse_pos = pygame.mouse.get_pos()

        for event in events:
            if self.reset_button.is_clicked(event):
                self.env.reset()
            if self.config_button.is_clicked(event):
                self.game_state = "CONFIG"
                return

        # Update UI and Game
        self.reset_button.update(mouse_pos)
        self.config_button.update(mouse_pos)

        action1, action2 = self.controller.get_actions()

        self.env.step([action1, action2])

        obs = self.env.render("rgb_array")

        obs_surface = pygame.surfarray.make_surface(obs.swapaxes(0, 1))

        scaled_surface = pygame.transform.scale(obs_surface, (SCREEN_WIDTH, SCREEN_HEIGHT))

        self.screen.blit(scaled_surface, (0, 0))

        self.reset_button.draw(self.screen)
        self.config_button.draw(self.screen)


    async def run(self):
        """The main application loop."""
        running = True
        while running:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    running = False
            
            if self.game_state == "CONFIG":
                self.run_config_state(events)
            elif self.game_state == "GAME":
                self.run_game_state(events)

            pygame.display.flip()
            self.clock.tick(self.config["fps"])
            await asyncio.sleep(0)

        pygame.quit()
        print("Exiting...")

# Main async function required by pygbag
async def main():
    print("Importing Pygame...")
    import pygame
    import numpy
    print("Import complete.")

    print("Initializing Pygame...")
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Moving Out - Configurable")
    print("Pygame initialized.")
    
    app = App(screen)
    await app.run()

# Entry point for local testing
if __name__ == "__main__":
    asyncio.run(main())
