"""A user interface for teleoperating an agent in an x-magical environment.

Modified from https://github.com/unixpickle/obs-tower2/blob/master/obs_tower2/recorder/env_interactor.py
"""


import time
from typing import List
import numpy as np
import pygame

from .utils import resultant_vector


class JoystickController:
    """User interface for interacting in an x-magical environment."""

    def __init__(
        self,
        resolution: int = 384,
        initial_angle=[0, 0],
    ):
        self.reset()
        self.last_angle = initial_angle

        self.trigger_threshold = 0.3
        self.reverse_action = False
        pygame.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            print("No joystick detected")
            # sys.exit()
        self.joysticks = []
        for i in range(joystick_count):
            joystick = pygame.joystick.Joystick(i)
            joystick.init()
            self.joysticks.append(joystick)
            print(f"Detected {i}: {joystick.get_name()}")

    def update_last_angle(self, angle, robot_id):
        self.last_angle[robot_id] = angle

    def get_actions(self) -> List[float]:
        self.joystick_event_process()
        x_displacement = self.axis[0][0]
        y_displacement = self.axis[0][1]

        if x_displacement != 0 or y_displacement != 0:
            if (
                np.linalg.norm((self.axis[0][0], self.axis[0][1]), ord=2)
                <= self.trigger_threshold
            ):
                magnitude = 0
                angle = self.last_angle[0]

            else:
                magnitude, angle = resultant_vector(
                    [[0, 0], [x_displacement, 0]], [[0, 0], [0, y_displacement]]
                )
                magnitude = (magnitude - self.trigger_threshold) / (
                    1 - self.trigger_threshold
                )
                magnitude = 1 if magnitude >= 1 else magnitude
                magnitude = -1 if magnitude <= -1 else magnitude
                if self.move_back[0]:
                    magnitude = -magnitude
                    angle = (angle + np.pi) % (2 * np.pi)
                if self.only_rotate[0]:
                    magnitude = 0

        else:
            magnitude = 0
            angle = self.last_angle[0]
        if self.reverse_action:
            action_1 = [magnitude, angle, self.hold[0]]
            self.update_last_angle(angle, 1)
        else:
            action_0 = [magnitude, angle, self.hold[0]]
            self.update_last_angle(angle, 0)

        x_displacement = self.axis[1][0]
        y_displacement = self.axis[1][1]
        if x_displacement != 0 or y_displacement != 0:
            if (
                np.linalg.norm((x_displacement, y_displacement), ord=2)
                <= self.trigger_threshold
            ):
                magnitude = 0
                angle = self.last_angle[1]

            else:
                magnitude, angle = resultant_vector(
                    [[0, 0], [x_displacement, 0]], [[0, 0], [0, y_displacement]]
                )
                magnitude = (magnitude - self.trigger_threshold) / (
                    1 - self.trigger_threshold
                )
                magnitude = 1 if magnitude >= 1 else magnitude
                magnitude = -1 if magnitude <= -1 else magnitude
                if self.move_back[1]:
                    magnitude = -magnitude
                    angle = (angle + np.pi) % (2 * np.pi)
                if self.only_rotate[1]:
                    magnitude = 0

        else:
            magnitude = 0
            angle = self.last_angle[1]
        if self.reverse_action:
            action_0 = [magnitude, angle, self.hold[1]]
            self.update_last_angle(angle, 0)
        else:
            action_1 = [magnitude, angle, self.hold[1]]
            self.update_last_angle(angle, 1)
        # action_1 = [magnitude, angle , self.hold[1]]
        # self.update_last_angle(angle, 1)

        action = [action_0, action_1]

        self.hold = [False, False]
        return action

    def reset(self):
        self._started = False
        self._finish_early = False
        self._last_image = None
        self.move_back = [False, False]
        self.only_rotate = [False, False]
        self.hold = [False, False]
        self.axis = [[0, 0, 0, 0], [0, 0, 0, 0]]
        self.start_moving = [False, False]

        self.action_hold_button = [5, 10]
        self.action_only_rotate_button = [4]

    def joystick_event_process(self):
        joystick_offset = [0.00, 0.9]
        # print(pygame.event.get())
        for event in pygame.event.get():
            # print(event)
            try:
                self.start_moving[event.joy] = True
            except:
                pass
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button in self.action_hold_button:
                    self.hold[event.joy] = True
                if event.button in self.action_only_rotate_button:
                    self.only_rotate[event.joy] = True
            elif event.type == pygame.JOYBUTTONUP:
                if event.button in self.action_hold_button:
                    self.hold[event.joy] = False
                if event.button in self.action_only_rotate_button:
                    self.only_rotate[event.joy] = False
            elif event.type == pygame.JOYAXISMOTION:
                if event.axis == 0:
                    if abs(event.value) < joystick_offset[0]:
                        self.axis[event.joy][1] = 0
                    elif (
                        abs(event.value) > joystick_offset[0]
                        and abs(event.value) < joystick_offset[1]
                    ):
                        self.axis[event.joy][1] = -(
                            event.value - joystick_offset[0]
                        ) / (joystick_offset[1] - joystick_offset[0])
                    elif abs(event.value) > joystick_offset[1]:
                        self.axis[event.joy][1] = -1 if (event.value > 0) else 1
                if event.axis == 1:
                    if abs(event.value) < joystick_offset[0]:
                        self.axis[event.joy][0] = 0
                    elif (
                        abs(event.value) > joystick_offset[0]
                        and abs(event.value) < joystick_offset[1]
                    ):
                        self.axis[event.joy][0] = -(
                            event.value - joystick_offset[0]
                        ) / (joystick_offset[1] - joystick_offset[0])
                    elif abs(event.value) > joystick_offset[1]:
                        self.axis[event.joy][0] = -1 if (event.value > 0) else 1
                if event.axis == 2:
                    if abs(event.value) < joystick_offset[0]:
                        self.axis[event.joy][2] = 0
                    elif (
                        abs(event.value) > joystick_offset[0]
                        and abs(event.value) < joystick_offset[1]
                    ):
                        self.axis[event.joy][2] = -(
                            event.value - joystick_offset[0]
                        ) / (joystick_offset[1] - joystick_offset[0])
                    elif abs(event.value) > joystick_offset[1]:
                        self.axis[event.joy][2] = -1 if (event.value > 0) else 1
                if event.axis == 3:
                    if abs(event.value) < joystick_offset[0]:
                        self.axis[event.joy][3] = 0
                    elif (
                        abs(event.value) > joystick_offset[0]
                        and abs(event.value) < joystick_offset[1]
                    ):
                        self.axis[event.joy][3] = -(
                            event.value - joystick_offset[0]
                        ) / (joystick_offset[1] - joystick_offset[0])
                    elif abs(event.value) > joystick_offset[1]:
                        self.axis[event.joy][3] = -1 if (event.value > 0) else 1

                if event.axis == 4:
                    if event.value > 0.5:
                        self.move_back[event.joy] = True
                    elif event.value < 0.5:
                        self.move_back[event.joy] = False
            elif event.type == pygame.JOYHATMOTION:
                pass



