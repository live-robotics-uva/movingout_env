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
import os
import time
# --- Constants and Configuration ---
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 700  # Increased height to accommodate status bar
GAME_AREA_HEIGHT = 600  # Height for the game environment
STATUS_BAR_HEIGHT = 100  # Height for the status bar at bottom
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

class ImageDropdownSelector:
    """Image dropdown selector that displays map images and names"""
    def __init__(self, x, y, width, options, font, image_size=(150, 150)):
        self.x = x
        self.y = y
        self.width = width
        self.options = options
        self.font = font
        self.image_size = image_size
        self.selected_option = options[0]
        self.is_expanded = False
        
        # Color definitions
        self.bg_color = (60, 60, 80)
        self.selected_color = (80, 180, 80)
        self.hover_color = (100, 100, 150)
        self.border_color = (150, 150, 150)
        
        # Calculate height - image + text + margins
        text_height = font.get_height()
        self.header_height = image_size[1] + text_height + 30  # Image height + text height + margins
        self.dropdown_item_height = 40
        
        # Define areas
        self.header_rect = pygame.Rect(x, y, width, self.header_height)
        
        # Load images
        self.images = {}
        self.load_images()
        
    def load_images(self):
        """Load all map images"""
        from moving_out.env_parameters import get_map_image_path
        
        for option in self.options:
            if(isinstance(option, int)):
                continue
            try:
                image_path = get_map_image_path(option)
                if os.path.exists(image_path):
                    image = pygame.image.load(image_path)
                    # Scale image to specified size
                    scaled_image = pygame.transform.scale(image, self.image_size)
                    self.images[option] = scaled_image
                else:
                    # If image doesn't exist, create a placeholder
                    placeholder = pygame.Surface(self.image_size)
                    placeholder.fill((128, 128, 128))
                    self.images[option] = placeholder
            except Exception as e:
                print(f"Failed to load image {option}: {e}")
                # Create placeholder
                placeholder = pygame.Surface(self.image_size)
                placeholder.fill((128, 128, 128))
                self.images[option] = placeholder
    
    def handle_events(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos
            
            # Click on header area
            if self.header_rect.collidepoint(mouse_pos):
                self.is_expanded = not self.is_expanded
                return
            
            # If dropdown is expanded, check if any option was clicked
            if self.is_expanded:
                dropdown_y = self.header_rect.bottom
                for i, option in enumerate(self.options):
                    option_rect = pygame.Rect(
                        self.x, 
                        dropdown_y + i * self.dropdown_item_height, 
                        self.width, 
                        self.dropdown_item_height
                    )
                    if option_rect.collidepoint(mouse_pos):
                        self.selected_option = option
                        self.is_expanded = False
                        print(f"Selected map: {self.selected_option}")
                        return
                
                # Click outside dropdown, close menu
                self.is_expanded = False
    
    def update(self, mouse_pos):
        # Can add hover effects here
        pass
    
    def draw(self, screen):
        # Draw main selection area
        pygame.draw.rect(screen, self.bg_color, self.header_rect, border_radius=8)
        pygame.draw.rect(screen, self.border_color, self.header_rect, 2, border_radius=8)
        
        # Draw selected map image and name
        if self.selected_option in self.images:
            image = self.images[self.selected_option]
            
            # Center the image
            image_x = self.header_rect.x + (self.header_rect.width - image.get_width()) // 2
            image_y = self.header_rect.y + 10  # Leave some margin at top
            screen.blit(image, (image_x, image_y))
            
            # Center the map name below the image
            text_surf = self.font.render(str(self.selected_option), True, (255, 255, 255))
            text_x = self.header_rect.x + (self.header_rect.width - text_surf.get_width()) // 2
            text_y = image_y + image.get_height() + 8  # Leave some space below image
            screen.blit(text_surf, (text_x, text_y))
        
        # Draw dropdown arrow (in bottom right corner)
        arrow_size = 8
        arrow_x = self.header_rect.right - 20
        arrow_y = self.header_rect.bottom - 15
        
        if self.is_expanded:
            # Up arrow
            arrow_points = [
                (arrow_x, arrow_y - arrow_size//2),
                (arrow_x - arrow_size//2, arrow_y + arrow_size//2),
                (arrow_x + arrow_size//2, arrow_y + arrow_size//2)
            ]
        else:
            # Down arrow
            arrow_points = [
                (arrow_x, arrow_y + arrow_size//2),
                (arrow_x - arrow_size//2, arrow_y - arrow_size//2),
                (arrow_x + arrow_size//2, arrow_y - arrow_size//2)
            ]
        
        pygame.draw.polygon(screen, (255, 255, 255), arrow_points)
        
        # If expanded, draw dropdown options
        if self.is_expanded:
            dropdown_y = self.header_rect.bottom
            dropdown_height = len(self.options) * self.dropdown_item_height
            dropdown_rect = pygame.Rect(self.x, dropdown_y, self.width, dropdown_height)
            
            # Draw dropdown background
            pygame.draw.rect(screen, self.bg_color, dropdown_rect, border_radius=8)
            pygame.draw.rect(screen, self.border_color, dropdown_rect, 2, border_radius=8)
            
            # Draw each option
            mouse_pos = pygame.mouse.get_pos()
            for i, option in enumerate(self.options):
                option_y = dropdown_y + i * self.dropdown_item_height
                option_rect = pygame.Rect(self.x, option_y, self.width, self.dropdown_item_height)
                
                # Mouse hover effect
                if option_rect.collidepoint(mouse_pos):
                    pygame.draw.rect(screen, self.hover_color, option_rect)
                
                # Highlight selected item
                if option == self.selected_option:
                    pygame.draw.rect(screen, (60, 150, 60, 100), option_rect)
                
                # Draw option text
                text_surf = self.font.render(str(option), True, (255, 255, 255))
                text_x = option_rect.x + 10
                text_y = option_rect.y + (option_rect.height - text_surf.get_height()) // 2
                screen.blit(text_surf, (text_x, text_y))

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
        self.game_start_time = None  # Track when game started
        self.init_config_ui()

    def init_config_ui(self):
        """Initializes all UI elements for the configuration screen."""
        self.font_title = pygame.font.Font(None, 48)
        self.font_label = pygame.font.Font(None, 36)
        self.font_option = pygame.font.Font(None, 32)

        # Map Selector - increased width for larger square images
        from moving_out.env_parameters import AVAILABLE_MAPS
        # Filter to only include string keys to avoid duplication
        self.map_options = [key for key in AVAILABLE_MAPS.keys() if isinstance(key, str)]
        self.config["map"] = self.map_options[0]
        self.map_selector = ImageDropdownSelector(50, 100, 200, self.map_options, self.font_option)

        # Controller Selector - adjusted position to avoid overlap
        self.controller_options = ["Keyboard", "Joystick"]
        self.controller_selector = OptionSelector(270, 100, self.controller_options, self.font_option)

        # FPS Input
        self.fps_input = TextInputBox(270, 350, 150, 40, self.font_option)
        
        # Start Button
        self.start_button = Button(200, 520, 200, 50, "Start Game", self.font_label, (50, 150, 50), (80, 180, 80))

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
        self.screen.blit(controller_label, (270, 60))
        self.controller_selector.draw(self.screen)

        fps_label = self.font_label.render("FPS", True, (255, 255, 255))
        self.screen.blit(fps_label, (270, 310))
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
            
        # Initialize game start time
        self.game_start_time = time.time()
            
        # UI for the game screen - positioned in status bar
        self.reset_button = Button(20, GAME_AREA_HEIGHT + 30, 100, 40, "Reset", self.font_option)
        self.config_button = Button(140, GAME_AREA_HEIGHT + 30, 80, 40, "Back", self.font_option)

    def run_game_state(self, events):
        """Handles logic for the main game screen."""
        mouse_pos = pygame.mouse.get_pos()

        for event in events:
            if self.reset_button.is_clicked(event):
                self.env.reset()
                self.game_start_time = time.time()  # Reset timer
            if self.config_button.is_clicked(event):
                self.game_state = "CONFIG"
                return

        # Update UI and Game
        self.reset_button.update(mouse_pos)
        self.config_button.update(mouse_pos)

        action1, action2 = self.controller.get_actions()

        self.env.step([action1, action2])

        # Render environment to upper area only
        obs = self.env.render("rgb_array")
        obs_surface = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
        scaled_surface = pygame.transform.scale(obs_surface, (SCREEN_WIDTH, GAME_AREA_HEIGHT))
        
        # Fill screen background
        self.screen.fill((40, 40, 60))
        
        # Draw game environment in upper area
        self.screen.blit(scaled_surface, (0, 0))
        
        # Draw status bar
        self.draw_status_bar()

    def draw_status_bar(self):
        """Draw the status bar with buttons, progress bar, and timer."""
        # Draw status bar background
        status_bar_rect = pygame.Rect(0, GAME_AREA_HEIGHT, SCREEN_WIDTH, STATUS_BAR_HEIGHT)
        pygame.draw.rect(self.screen, (30, 30, 50), status_bar_rect)
        pygame.draw.line(self.screen, (100, 100, 120), 
                        (0, GAME_AREA_HEIGHT), (SCREEN_WIDTH, GAME_AREA_HEIGHT), 2)
        
        # Draw buttons
        self.reset_button.draw(self.screen)
        self.config_button.draw(self.screen)
        
        # Get global score and draw progress bar
        try:
            global_score = self.env.global_score()
        except:
            global_score = 0.0
            
        # Progress bar
        progress_bar_x = 250
        progress_bar_y = GAME_AREA_HEIGHT + 40
        progress_bar_width = 200
        progress_bar_height = 20
        
        # Background of progress bar
        progress_bg_rect = pygame.Rect(progress_bar_x, progress_bar_y, progress_bar_width, progress_bar_height)
        pygame.draw.rect(self.screen, (60, 60, 80), progress_bg_rect)
        pygame.draw.rect(self.screen, (150, 150, 150), progress_bg_rect, 2)
        
        # Progress fill
        progress_fill_width = int(progress_bar_width * global_score)
        if progress_fill_width > 0:
            progress_fill_rect = pygame.Rect(progress_bar_x, progress_bar_y, progress_fill_width, progress_bar_height)
            # Green gradient for progress
            color_intensity = min(255, int(150 + global_score * 105))
            pygame.draw.rect(self.screen, (0, color_intensity, 0), progress_fill_rect)
        
        # Progress text - centered above progress bar
        progress_text = f"Score: {global_score:.1%}"
        progress_surf = self.font_option.render(progress_text, True, (255, 255, 255))
        # Center the text horizontally above the progress bar
        text_x = progress_bar_x + (progress_bar_width - progress_surf.get_width()) // 2
        text_y = GAME_AREA_HEIGHT + 15
        self.screen.blit(progress_surf, (text_x, text_y))
        
        # Calculate and display elapsed time (simplified format)
        if self.game_start_time:
            elapsed_time = time.time() - self.game_start_time
            time_text = f"{elapsed_time:.1f}s"
        else:
            time_text = "0.0s"
            
        time_surf = self.font_option.render(time_text, True, (255, 255, 255))
        time_x = SCREEN_WIDTH - time_surf.get_width() - 20
        time_y = GAME_AREA_HEIGHT + 40
        self.screen.blit(time_surf, (time_x, time_y))


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
    pygame.display.set_caption("Moving Out - Game with Status Bar")
    print("Pygame initialized.")
    
    app = App(screen)
    await app.run()

# Entry point for local testing
if __name__ == "__main__":
    asyncio.run(main())