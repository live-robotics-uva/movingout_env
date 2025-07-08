import math
from typing import Tuple, Union

import numpy as np
import pymunk as pm
from moving_out import geom as gtools
from moving_out import render as r
from moving_out.env_parameters import (COLORS_RGB, ROBOT_LINE_THICKNESS,
                                       darken_rgb, lighten_rgb)

from .base import NonHolonomicEmbodiment


class NonHolonomicRoundEmbodiment(NonHolonomicEmbodiment):
    """A non-holonomic embodiment with fingers that open and close."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if self.id == 1:
            self.robot_color = COLORS_RGB["light_blue"]
            self.robot_hold_color = COLORS_RGB["less_light_blue"]
        if self.id == 2:
            self.robot_color = COLORS_RGB["light_red"]
            self.robot_hold_color = COLORS_RGB["less_light_red"]

        self.eye_txty = (0.4 * self.radius, 0.3 * self.radius)
        self.hold = False
        self.holded_item = None

    def revert_color(self):
        if self.id == 2:
            self.robot_color = COLORS_RGB["light_blue"]
            self.robot_hold_color = COLORS_RGB["less_light_blue"]

        if self.id == 1:
            self.robot_color = COLORS_RGB["light_red"]
            self.robot_hold_color = COLORS_RGB["less_light_red"]
        dark_robot_color = darken_rgb(self.robot_color)
        self.graphic_bodies[0].outline_color = dark_robot_color

    def revert_back(self):
        if self.id == 1:
            self.robot_color = COLORS_RGB["light_blue"]
            self.robot_hold_color = COLORS_RGB["less_light_blue"]
        if self.id == 2:
            self.robot_color = COLORS_RGB["light_red"]
            self.robot_hold_color = COLORS_RGB["less_light_red"]
        dark_robot_color = darken_rgb(self.robot_color)
        self.graphic_bodies[0].outline_color = dark_robot_color

    def _setup_body(self):
        inertia = pm.moment_for_circle(self.mass, 0, self.radius, (0, 0))
        self.body = pm.Body(self.mass, inertia)

    def _setup_shape(self):
        friction = 0.5
        self.shape = pm.Circle(self.body, self.radius, (0, 0))
        self.shape.friction = friction

    def _setup_graphic(self):
        graphics_body = r.make_circle(self.radius, 100, True)
        dark_robot_color = darken_rgb(self.robot_color)
        graphics_body.color = self.robot_color
        graphics_body.outline_color = dark_robot_color
        self.graphic_bodies.append(graphics_body)

    def set_action(
        self,
        action: Union[np.ndarray, Tuple[float, float, float]],
    ) -> None:
        super().set_action(action[:4])

    def update(self, dt: float) -> None:
        super().update(dt)

    def pre_draw(self) -> None:
        super().pre_draw()

        if self.hold:
            self.graphic_bodies[0].color = self.robot_hold_color
        else:
            self.graphic_bodies[0].color = self.robot_color
