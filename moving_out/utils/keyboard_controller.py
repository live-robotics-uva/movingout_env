import math
import pygame


class KeyboardController:
    """
    A class to handle keyboard inputs for two players and map them to specific
    action spaces for the MovingOutEnv.
    Action Space: [forward, angle, grab]
    """

    def __init__(self):
        """
        Initializes the controller and defines the mapping from key combinations
        to directional angles.
        """
        self.key_to_angle_map = {
            # Tuple format: (Up, Left, Down, Right)
            (False, False, False, True): -math.pi / 2,  # Right
            (True, False, False, True): -math.pi / 4,  # Up-Right
            (True, False, False, False): 0,  # Up
            (True, True, False, False): math.pi / 4,  # Up-Left
            (False, True, False, False): math.pi / 2,  # Left
            (False, True, True, False): 3 * math.pi / 4,  # Down-Left
            (False, False, True, False): math.pi,  # Down
            (False, False, True, True): -3 * math.pi / 4,  # Down-Right
        }
        # --- ADDED: State trackers for grab keys ---
        # These will track if the key was pressed in the *previous* frame.
        self.p1_grab_pressed_last_frame = False
        self.p2_grab_pressed_last_frame = False

    def get_actions(self):
        """
        Gets the current state of the keyboard and returns a tuple of two
        actions, one for each player. This should be called once per frame.

        :return: A tuple (action1, action2)
        """
        # Get the state of all keyboard buttons at once for efficiency.
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        # print(keys)
        # --- Process Action 1 (WASD + Space) ---
        w = keys[pygame.K_w]
        a = keys[pygame.K_a]
        s = keys[pygame.K_s]
        d = keys[pygame.K_d]
        p1_grab_is_pressed = keys[pygame.K_SPACE]

        p1_is_moving = any((w, a, s, d))
        action1_forward = 1 if p1_is_moving else 0

        direction_tuple_1 = (w, a, s, d) # Note: For WASD, tuple order should be (w, a, s, d)
        action1_angle = self.key_to_angle_map.get(direction_tuple_1, 0)

        # --- MODIFIED: Logic for single-shot grab ---
        action1_grab = 0
        # Check if the key is pressed NOW and was NOT pressed BEFORE.
        if p1_grab_is_pressed and not self.p1_grab_pressed_last_frame:
            action1_grab = 1
        # Update the state for the next frame.
        self.p1_grab_pressed_last_frame = p1_grab_is_pressed

        action1 = [action1_forward, action1_angle, action1_grab]

        # --- Process Action 2 (Arrow Keys + Enter) ---
        up = keys[pygame.K_UP]
        left = keys[pygame.K_LEFT]
        down = keys[pygame.K_DOWN]
        right = keys[pygame.K_RIGHT]
        p2_grab_is_pressed = keys[pygame.K_RETURN] or keys[pygame.K_KP_ENTER]

        p2_is_moving = any((up, left, down, right))
        action2_forward = 1 if p2_is_moving else 0

        direction_tuple_2 = (up, left, down, right)
        action2_angle = self.key_to_angle_map.get(direction_tuple_2, 0)

        # --- MODIFIED: Logic for single-shot grab ---
        action2_grab = 0
        # Check if the key is pressed NOW and was NOT pressed BEFORE.
        if p2_grab_is_pressed and not self.p2_grab_pressed_last_frame:
            action2_grab = 1
        # Update the state for the next frame.
        self.p2_grab_pressed_last_frame = p2_grab_is_pressed

        action2 = [action2_forward, action2_angle, action2_grab]

        return action1, action2