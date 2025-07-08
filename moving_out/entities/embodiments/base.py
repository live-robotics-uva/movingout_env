import abc
import math
from typing import Tuple, Union

import numpy as np
import pymunk as pm
from moving_out import render as r
from moving_out.entities.base import Entity

# pytype: disable=attribute-error


class Embodiment(Entity, abc.ABC):
    """Base abstraction for robotic embodiments."""

    def __init__(
        self,
        radius: float,
        init_pos,
        init_angle: float,
        mass: float = 100.0,
        id: int = 0,
    ) -> None:
        self.radius = radius
        self.init_pos = init_pos
        self.init_angle = init_angle
        self.mass = mass
        self.id = id

        # These need to be constructed in the _setup methods below.
        self.control_body = None
        self.body = None
        self.shape = None
        self.graphic_bodies = []
        self.xform = None

        # These do not necessarily need to be constructed.
        self.extra_bodies = []
        self.extra_shapes = []
        self.extra_graphic_bodies = []
        self.target_angle = 0
        self.x_displacement = 0
        self.y_displacement = 0
        self.x_rotation_displacement = 0
        self.y_rotation_displacement = 0

    @abc.abstractclassmethod
    def _setup_body(self):
        pass

    def _setup_extra_bodies(self):
        pass

    @abc.abstractclassmethod
    def _setup_control_body(self):
        pass

    @abc.abstractclassmethod
    def _setup_shape(self):
        pass

    def _setup_extra_shapes(self):
        pass

    @abc.abstractclassmethod
    def _setup_graphic(self):
        pass

    def _setup_extra_graphics(self):
        pass

    @abc.abstractclassmethod
    def set_action(self, move_action: np.ndarray) -> None:
        pass

    @abc.abstractclassmethod
    def pre_draw(self) -> None:
        pass

    @abc.abstractclassmethod
    def update(self, dt: float) -> None:
        pass

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)

        self._setup_body()
        assert self.body is not None
        self.body.position = self.init_pos
        self.body.angle = self.init_angle
        self.add_to_space(self.body)

        # Main body position and angle should be set before calling setup on the
        # extra bodies since they might depend on those values.
        self._setup_extra_bodies()
        self.add_to_space(*self.extra_bodies)

        self._setup_control_body()
        assert self.control_body is not None

        self._setup_shape()
        self._setup_extra_shapes()
        assert self.shape is not None
        self.add_to_space(self.shape)
        self.add_to_space(self.extra_shapes)

        self._setup_graphic()
        self._setup_extra_graphics()
        assert self.graphic_bodies

        self.xform = r.Transform()
        self.robot_compound = r.Compound(
            [*self.graphic_bodies, *self.extra_graphic_bodies]
        )
        self.robot_compound.add_transform(self.xform)
        self.viewer.add_geom(self.robot_compound)


class NonHolonomicEmbodiment(Embodiment):
    """A embodiment with 2 degrees of freedom: velocity and turning angle."""

    DOF = 3  # Degrees of freedom.

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.rel_turn_angle = 0.0
        self.target_speed = 0.0
        self._speed_limit = 4.0 * self.radius
        self._angle_limit = 2
        self.if_control_angle = True
        self.eye_txty = (0.4 * self.radius, 0.0 * self.radius)
        self.started = False
        self.slow_movement = False
        self.small_items_angle = False
        self.move_together = False

    def reconstruct_signature(self):
        kwargs = dict(
            radius=self.radius,
            init_pos=self.body.position,
            init_angle=self.body.angle,
            mass=self.mass,
        )
        return type(self), kwargs

    def _setup_control_body(self):
        self.control_body = control_body = pm.Body(body_type=pm.Body.KINEMATIC)
        control_body.position = self.init_pos
        control_body.angle = self.init_angle
        self.add_to_space(control_body)
        pos_control_joint = pm.PivotJoint(control_body, self.body, (0, 0), (0, 0))
        pos_control_joint.max_bias = 0
        pos_control_joint.max_force = self.phys_vars.robot_pos_joint_max_force
        self.add_to_space(pos_control_joint)
        rot_control_joint = pm.GearJoint(control_body, self.body, 0.0, 1.0)
        rot_control_joint.error_bias = 0.0
        rot_control_joint.max_bias = 5
        rot_control_joint.max_force = self.phys_vars.robot_rot_joint_max_force
        self.add_to_space(rot_control_joint)

    def _setup_extra_bodies(self):
        # Googly eye control bodies & joints.
        self.pupil_bodies = []
        for _ in range(2):
            eye_mass = self.mass / 10
            eye_radius = self.radius
            eye_inertia = pm.moment_for_circle(eye_mass, 0, eye_radius, (0, 0))
            eye_body = pm.Body(eye_mass, eye_inertia)
            eye_body.angle = self.init_angle
            eye_joint = pm.DampedRotarySpring(self.body, eye_body, 0, 0.1, 3e-3)
            eye_joint.max_bias = 3.0
            eye_joint.max_force = 0.001
            self.pupil_bodies.append(eye_body)
            self.add_to_space(eye_joint)
        self.extra_bodies.extend(self.pupil_bodies)

    def _setup_extra_graphics(self):
        self.eye_shapes = []
        self.pupil_transforms = []
        for x_sign in [-1, 1]:
            eye = r.make_circle(0.2 * self.radius, 100, outline=False)
            eye.color = (1.0, 1.0, 1.0)  # White color.
            eye_base_transform = r.Transform(
                translation=(x_sign * self.eye_txty[0], self.eye_txty[1])
            )
            eye.add_transform(eye_base_transform)
            pupil = r.make_circle(0.12 * self.radius, 100, outline=False)
            pupil.color = (0.1, 0.1, 0.1)  # Black color.
            pupil_transform = r.Transform()
            pupil.add_transform(r.Transform(translation=(0, self.radius * 0.07)))
            pupil.add_transform(pupil_transform)
            pupil.add_transform(eye_base_transform)
            self.pupil_transforms.append(pupil_transform)
            self.eye_shapes.extend([eye, pupil])
        self.extra_graphic_bodies.extend(self.eye_shapes)

    def set_action(self, action: Union[np.ndarray, Tuple[float, float]]) -> None:
        # assert len(action) == NonHolonomicEmbodiment.DOF
        self.started = True
        self.action_dim = len(action)

        current_diff = self.control_body.angle - self.body.angle
        self.control_body.angle = self.body.angle + np.clip(current_diff, -2, 2)

        magnitude = action[0]
        angle = action[1]

        if magnitude < 0:
            x_displacement = magnitude * np.cos(angle)
            y_displacement = magnitude * np.sin(angle)
        else:
            x_displacement = magnitude * np.cos(angle)
            y_displacement = magnitude * np.sin(angle)
        # if(magnitude < 0):
        #     x_displacement = -x_displacement
        #     y_displacement = -y_displacement
        if self.slow_movement:
            self.x_displacement = x_displacement * 0.1
            self.y_displacement = y_displacement * 0.1
        else:
            self.x_displacement = x_displacement
            self.y_displacement = y_displacement

        self.target_angle = angle

        if x_displacement == 0 and y_displacement == 0:
            self.target_angle = self.control_body.angle
            magnitude = 0

        current_angle = self.body.angle
        self.angle_diff = (self.target_angle - current_angle) % (2 * math.pi)
        self.angle_diff = (self.angle_diff + math.pi) % (2 * math.pi) - math.pi
        if (
            not self.if_control_angle
            and not self.move_together
            and not self.small_items_angle
        ):
            # self.angle_diff = self.angle_diff / 50
            factor = 0.5
            if abs(self.angle_diff) > np.pi * factor:
                if self.angle_diff > 0:
                    self.angle_diff = np.pi - self.angle_diff
                elif self.angle_diff < 0:
                    self.angle_diff = -np.pi - self.angle_diff
                self.angle_diff = -self.angle_diff
                # if(self.move_large_item_together):
                #     self.angle_diff = self.angle_diff
                # else:
                #     self.angle_diff = -self.angle_diff
        elif self.move_together:
            factor = 0.5
            if abs(self.angle_diff) > np.pi * factor:
                if self.angle_diff > 0:
                    self.angle_diff = np.pi - self.angle_diff
                elif self.angle_diff < 0:
                    self.angle_diff = -np.pi - self.angle_diff
            self.angle_diff = -self.angle_diff / 10
        if self.slow_movement:
            self.angle_diff = self.angle_diff / 100
            self.slow_movement = False
            # self.angle_diff = -self.angle_diff

        # if(abs(self.angle_diff) < 0.1):
        #     self.angle_diff = 0
        # if(abs(self.angle_diff) > np.pi):
        #     if(self.angle_diff > 0):
        #         self.target_angle = self.target_angle - 2 * np.pi
        #         self.angle_diff = self.target_angle - current_angle
        #     elif(self.angle_diff < 0):
        #         self.target_angle = self.target_angle + 2 * np.pi
        #         self.angle_diff = self.target_angle - current_angle
        # if abs(self.angle_diff) > self._angle_limit * 0.3:
        #     angle_step = math.copysign(self._angle_limit * 0.3, self.angle_diff)
        #     self.control_body.angle += angle_step * 0.3

        # self.target_speed = np.clip(magnitude, -self._speed_limit, self._speed_limit)
        self.rel_turn_angle = np.clip(
            self.angle_diff, -self._angle_limit, self._angle_limit
        )
        if not self.hold or self.small_items_angle:
            if abs(self.rel_turn_angle) >= 0.4:
                self.x_displacement = x_displacement * 0.1
                self.y_displacement = y_displacement * 0.1
        self.target_angle = self.rel_turn_angle + self.body.angle

    def update(self, dt: float) -> None:
        if not self.started:
            return
        # del dt
        # self.control_body.angle = self.body.angle + self.rel_turn_angle

        # if abs(self.angle_diff) > self._angle_limit * 0.3:
        #     angle_step = math.copysign(self._angle_limit * 0.3, self.angle_diff)
        #     self.control_body.angle += angle_step * 0.3
        # else:
        #     self.control_body.angle = self.target_angle
        # self.control_body.angle = self.target_angle
        if (abs(self.y_displacement) + abs(self.x_displacement)) < 0.01:
            y_displacement = 0
            x_displacement = 0
            # self.control_body.angle = self.body.angle
        else:
            x_displacement = self.x_displacement
            y_displacement = self.y_displacement
            # if(self.if_control_angle):
        if self.action_dim == 4:
            if (
                abs(self.y_rotation_displacement) + abs(self.x_rotation_displacement)
            ) < 0.01:
                self.control_body.angle = self.body.angle
            else:
                self.control_body.angle = self.target_angle
        else:
            self.control_body.angle = self.target_angle
        # self.control_body.angle = self.abs_turn_angle
        x_vel_vector = pm.vec2d.Vec2d(-y_displacement * 0.6, +x_displacement * 0.6)
        # vel_vector = self.body.rotation_vector.cpvrotate(x_vel_vector)
        self.control_body.velocity = x_vel_vector

    def pre_draw(self) -> None:
        self.xform.reset(translation=self.body.position, rotation=self.body.angle)
        for pupil_xform, pupil_body in zip(self.pupil_transforms, self.pupil_bodies):
            pupil_xform.reset(rotation=pupil_body.angle - self.body.angle)
