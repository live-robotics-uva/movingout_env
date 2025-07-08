import abc
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import moving_out.entities as en
import moving_out.render as r
import numpy as np
import pymunk as pm
import pygame
from moving_out.entities.embodiments.round import NonHolonomicRoundEmbodiment
from moving_out.env_parameters import (ARENA_ZOOM_OUT, COLORS_RGB, PHYS_ITER,
                                       PHYS_STEPS, ROBOT_MASS, ROBOT_RAD,
                                       lighten_rgb)
from moving_out.utils.states_encoding import StatesEncoder
from moving_out.utils.utils import reset_env_to_id



# from gymnasium import spaces



class PhysicsVariables:
    """Simple physics variables class for environment configuration."""

    def __init__(self):
        # Default physics parameters
        self.robot_pos_joint_max_force = 1000.0
        self.robot_rot_joint_max_force = 100.0
        self.robot_finger_max_force = 50.0
        self.shape_trans_joint_max_force = 1000.0
        self.shape_rot_joint_max_force = 100.0

    @classmethod
    def defaults(cls):
        """Return default physics variables."""
        return cls()

    @classmethod
    def sample(cls, rng):
        """Sample physics variables with some randomness."""
        # For now, just return defaults
        # In the future, this could add random variations
        return cls.defaults()


class BaseEnv(abc.ABC):
    def __init__(
        self,
        *,  # Subclasses can have additional args.
        robot_0_cls: Type[
            en.embodiments.NonHolonomicEmbodiment
        ] = NonHolonomicRoundEmbodiment,
        robot_1_cls: Type[
            en.embodiments.NonHolonomicEmbodiment
        ] = NonHolonomicRoundEmbodiment,
        res_hw: Tuple[int, int] = (512, 512),
        max_episode_steps: Optional[int] = None,
        rand_dynamics: bool = False,
        arena_size=[1.2, 1.2],
    ) -> None:

        view_mode = "allo"
        self.ARENA_BOUNDS_LRBT = [
            -arena_size[0],
            arena_size[1],
            -arena_size[0],
            arena_size[1],
        ]
        self.ARENA_SIZE_MAX = max(self.ARENA_BOUNDS_LRBT)

        self.robot_0_cls = robot_0_cls
        self.action_0_dim = robot_0_cls.DOF
        self.robot_1_cls = robot_1_cls
        self.action_1_dim = robot_1_cls.DOF
        self.phys_iter = PHYS_ITER
        self.phys_steps = PHYS_STEPS
        self.res_hw = res_hw
        self.max_episode_steps = max_episode_steps
        self.rand_dynamics = rand_dynamics

        self.target_areas = None

        # State/rendering (see reset()).
        self._entities = None
        self._space = None
        self._robots = []
        self.joint = None
        self._episode_steps = None
        self._phys_vars = None
        self._renderer_func = (
            self._use_allo_cam
        )
        self.if_hold = [False, False]
        self.hold_joints = [None, None]
        self.motor_joints = [None, None]
        self.two_robot_joints = None
        self.shape_in_front = [None, None]
        # This is for rendering and displaying.
        self.renderer = None
        self.viewer = None

        self.slowly_movable_mass = 0.4

        # Set observation and action spaces.
        # self.observation_space = spaces.Box(
        #     low=0, high=255, shape=(*self.res_hw, 3), dtype=np.uint8
        # )
        # self.action_space = spaces.Box(
        #     np.array(
        #         [-1] * self.action_0_dim + [-1] * self.action_1_dim, dtype=np.float32
        #     ),
        #     np.array(
        #         [+1] * self.action_0_dim + [+1] * self.action_1_dim, dtype=np.float32
        #     ),
        #     dtype=np.float32,
        # )

        self.seed()
        self.states_convetor = StatesEncoder()

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Initialise the PRNG and return seed necessary to reproduce results.

        The action space should probably be seeded in a downstream RL
        application.
        """
        if seed is None:
            seed = np.random.randint(0, (1 << 31) - 1)
        self.rng = np.random.RandomState(seed=seed)
        return [seed]

    def _make_robot(
        self, init_pos: Union[np.ndarray, Tuple[float, float]], init_angle: float, id=1
    ) -> en.embodiments.NonHolonomicRoundEmbodiment:
        return self.robot_0_cls(
            radius=ROBOT_RAD,
            mass=ROBOT_MASS,
            init_pos=init_pos,
            init_angle=init_angle,
            id=id,
        )

    def _make_shape(self, **kwargs) -> en.Shape:
        return en.Shape(**kwargs)

    @abc.abstractmethod
    def on_reset(self) -> None:
        """Set up entities necessary for this environment, and reset any other
        data needed for the env. Must create a robot in addition to any
        necessary entities.
        """
        pass

    def add_entities(self, entities: Sequence[en.Entity]) -> None:
        """Adds a list of entities to the current entities list and sets it up.

        Only intended to be used from within on_reset(). Needs to be called for
        every created entity or else they will not be added to the space!
        """
        for i, entity in enumerate(entities):
            if isinstance(entity, self.robot_0_cls) or isinstance(
                entity, self.robot_1_cls
            ):
                self._robots.append(entity)
            self._entities.append(entity)
            entity.setup(self.renderer, self._space, self._phys_vars)

    # def _use_ego_cam(self) -> None:
    #     """Egocentric agent view."""
    #     self.renderer.set_cam_follow(
    #         source_xy_world=(
    #             self._robot.body.position.x,
    #             self._robot.body.position.y,
    #         ),
    #         target_xy_01=(0.5, 0.15),
    #         viewport_hw_world=(
    #             self._arena_h * ARENA_ZOOM_OUT,
    #             self._arena_w * ARENA_ZOOM_OUT,
    #         ),
    #         rotation=self._robot.body.angle,
    #     )

    def _use_allo_cam(self) -> None:
        """Allocentric 'god-mode' view."""
        self.renderer.set_bounds(
            left=self._arena.left * ARENA_ZOOM_OUT,
            right=self._arena.right * ARENA_ZOOM_OUT,
            bottom=self._arena.bottom * ARENA_ZOOM_OUT,
            top=self._arena.top * ARENA_ZOOM_OUT,
        )

    def get_encoded_state(self):
        raw_state = self.get_all_states()
        encoded_states = self.states_convetor.get_state_by_current_obs_states(raw_state)
        return encoded_states

    def reset(self, map_name=None, add_noise_to_item=None):
        if map_name is not None:
            self.map_name = map_name
        self._episode_steps = 0

        # Delete old entities/space.
        self._entities = []
        self._space = None
        self._robots = []

        self.if_hold = [False, False]
        self.hold_joints = [None, None]
        self.motor_joints = [None, None]
        self.two_robot_joints = None
        self.shape_in_front = [None, None]

        # self._robot_2 = None
        self._phys_vars = None
        self.joint = None

        if self.renderer is None:
            res_h, res_w = self.res_hw
            background_color = lighten_rgb(COLORS_RGB["grey"], times=4)

            # self.renderer = pm.Space()
            self.renderer = r.Viewer(res_w, res_h, background_color)
        else:
            # These will get added back later.
            self.renderer.reset_geoms()

        self._space = pm.Space()
        self._space.collision_slop = 0.001
        self._space.iterations = self.phys_iter

        if self.rand_dynamics:
            # Randomise the physics properties of objects and the robot a
            # little bit.
            self._phys_vars = PhysicsVariables.sample(self.rng)
        else:
            self._phys_vars = PhysicsVariables.defaults()

        # Set up robot and arena.
        arena_l, arena_r, arena_b, arena_t = self.ARENA_BOUNDS_LRBT
        self._arena = en.ArenaBoundaries(
            left=arena_l, right=arena_r, bottom=arena_b, top=arena_t
        )
        self._arena_w = arena_r - arena_l
        self._arena_h = arena_t - arena_b
        self.add_entities([self._arena])

        # reset_rv = self.on_reset()
        # assert reset_rv is None, (
        # f"on_reset method of {type(self)} returned {reset_rv}, but "
        # f"should return None"
        # )
        # assert isinstance(self._robot, self.robot_0_cls)
        # assert len(self._entities) >= 1

        assert np.allclose(self._arena.left + self._arena.right, 0)
        assert np.allclose(self._arena.bottom + self._arena.top, 0)

        self._renderer_func()

        reset_env_to_id(self, self.map_name, add_noise_to_item=add_noise_to_item)
        # return self.render(mode="rgb_array")
        obs = self.get_encoded_state()
        info = {}
        return obs, info

    def _phys_steps_on_frame(self):
        spf = 1 / 20.0
        dt = spf / self.phys_steps
        for i in range(self.phys_steps):
            for ent in self._entities:
                ent.update(dt)
            self._space.step(dt)

    @abc.abstractmethod
    def global_score(self) -> float:
        """Compute the score for this trajectory.

        Only called at the last step of the trajectory.

        Returns:
           score: number in [0, 1] indicating the worst possible
               performance (0), the best possible performance (1) or something
               in between. Should apply to the WHOLE trajectory.
        """
        pass  # pytype: disable=bad-return-type

    @abc.abstractclassmethod
    def get_reward(self) -> float:
        """Compute the reward for the current timestep.

        This is called at the end of every timestep.
        """
        pass  # pytype: disable=bad-return-type

    def pre_solve(self, constraint, space):
        # This callback is called when the two bodies connected by the pin joint collide
        # body_a, body_b = arbiter.shapes
        # print(f"Collision detected between Body A and Body B")

        # You can perform additional logic here, such as adjusting the joint, applying forces, etc.

        return True

    def post_solve(self, constraint, space):
        # This callback is called when the two bodies connected by the pin joint collide
        # body_a, body_b = arbiter.shapes
        # print(f"Collision detected between Body A and Body B")

        # You can perform additional logic here, such as adjusting the joint, applying forces, etc.

        return True

    def get_direction_from_A_to_B(self, A, B):
        A = np.array(A)
        B = np.array(B)
        direction = B - A
        return direction

    def get_direction_from_agents_to_B(self, B):
        B = np.array(B)

        direction = []
        for i in range(2):
            direction.append(
                self.get_direction_from_A_to_B(self._robots[i].body.position, B)
            )
        return direction

    def update_robot_grab_item(self, robot, robot_id, shape_in_front):
        robot.hold = True
        robot.holded_item = shape_in_front
        shape_in_front.body.hold[robot_id] = True

    def update_robot_release_item(self, robot, robot_id):
        robot.hold = False
        robot.holded_item = None

    def grab_front(self, shape_in_front, robot, robot_id):
        # shape_in_front.body._BodyType = pm.Body.KINEMATIC
        # shape_in_front.body.moment = pm.inf
        self.hold_joints[robot_id] = pm.PivotJoint(
            robot.body, shape_in_front.body, robot.body.position
        )
        # rot_control_joint = pm.SimpleMotor(robot.body, shape_in_front.body, 0)
        rot_control_joint = pm.GearJoint(
            robot.body,
            shape_in_front.body,
            -robot.body.angle + shape_in_front.body.angle,
            1,
        )
        self.motor_joints[robot_id] = rot_control_joint

        def pre_solve(constraint, space):
            # print("PRE SLOVE")
            pass

        def post_solve(constraint, space):
            # print("POST SLOVE")
            pass

        self.hold_joints[robot_id].post_solve = post_solve
        self.hold_joints[robot_id].pre_solve = pre_solve

        self.update_robot_grab_item(robot, robot_id, shape_in_front)
        self.hold_joints[robot_id].error_bias = 0.0
        # self.joint_2 = pm.RotaryLimitJoint(self._robot.body, shape_in_front.body, 3.14 / 8, 0)
        # self.joint_2 = pm.DampedSpring(self._robot.body, shape_in_front.body,(10, 10), (-10, 10), 0, 10000000000, 10000)
        # self.joint_2.error_bias = 0.0
        self._space.add(self.hold_joints[robot_id])
        self._space.add(self.motor_joints[robot_id])
        if self._robots[robot_id].holded_item.body.shape_category == "small":
            self._robots[robot_id].if_control_angle = False
            self.set_body_movable(shape_in_front.body)
        if self._robots[robot_id].holded_item.body.shape_category == "middle":
            self.set_body_part_movable(shape_in_front.body)
            self._robots[robot_id].if_control_angle = False

        if (
            self.if_hold[1 - robot_id]
            and self._robots[1 - robot_id].holded_item.body.id
            == self._robots[robot_id].holded_item.body.id
        ):
            Pivot_point = (
                np.array(
                    self._robots[1 - robot_id].body.position
                    + self._robots[robot_id].body.position
                )
                / 2.0
            )
            # jointt = pm.PivotJoint(
            #     robot.body, self._robots[1-robot_id].body, list(Pivot_point)
            # )
            # self.two_robot_joints = jointt
            # self._space.add(jointt)
            self._robots[0].if_control_angle = False
            self._robots[1].if_control_angle = False
            self.set_body_movable(shape_in_front.body)
        if self._robots[robot_id].holded_item.body.shape_category == "small":
            self.set_body_movable(shape_in_front.body)
        # self._space.add(self.joint_2)

    def realease_front(self, robot, robot_id):
        if self._robots[robot_id].holded_item.body.shape_category == "middle":
            self.set_body_unmovable(self._robots[robot_id].holded_item.body)
            self._robots[robot_id].if_control_angle = True
        if self._robots[robot_id].holded_item.body.shape_category == "small":
            self._robots[robot_id].if_control_angle = True
            self.set_body_unmovable(self._robots[robot_id].holded_item.body)
        if self._robots[robot_id].holded_item.body.shape_category == "large":
            self._robots[robot_id].if_control_angle = True
        if (
            sum(self.if_hold) == 2
            and self._robots[1 - robot_id].holded_item.body.id
            == self._robots[robot_id].holded_item.body.id
        ):
            # self._space.remove(self.two_robot_joints)
            # self.two_robot_joints = None
            self._robots[robot_id].if_control_angle = True
            # self._robots[1].if_control_angle = True
            self.set_body_unmovable(self._robots[robot_id].holded_item.body)
        if (
            sum(self.if_hold) == 2
            and self._robots[1 - robot_id].holded_item.body.id
            == self._robots[robot_id].holded_item.body.id
            and self._robots[robot_id].holded_item.body.shape_category == "middle"
        ):
            self.set_body_part_movable(self._robots[1 - robot_id].holded_item.body)

        self._space.remove(self.hold_joints[robot_id])
        self.hold_joints[robot_id] = None
        self._space.remove(self.motor_joints[robot_id])
        self.motor_joints[robot_id] = None
        # self.set_body_unmovable(self._robots[robot_id].holded_item.body)
        self._robots[robot_id].holded_item.body.hold[robot_id] = False
        self._robots[robot_id].holded_item = None
        robot.hold = False

    def check_for_object_in_front(self, space, robot, distance=0.095):
        """Check if there is an object in front of the player within a certain distance"""
        player_body = robot.body
        player_pos = player_body.position
        angle = player_body.angle
        direction = pm.Vec2d(-math.sin(angle), math.cos(angle))
        query_pos = player_pos + direction * distance
        # Query for shapes in the given position
        shapes = space.point_query(query_pos, 0.04, pm.ShapeFilter(group=1))
        shapes = sorted(shapes, key=lambda x: x.distance)

        def are_perpendicular(angle1, angle2, tolerance=0.2):
            def normalize_angle(angle):
                return angle % (2 * math.pi)

            angle1 = normalize_angle(angle1)
            angle2 = normalize_angle(angle2)

            angle_diff = normalize_angle(angle2 - angle1)

            if angle_diff > math.pi:
                angle_diff -= 2 * math.pi

            return math.isclose(abs(angle_diff), math.pi / 2, abs_tol=tolerance)

        for shape in shapes:
            try:
                if shape.shape.body.pickable:
                    if (
                        shape.shape.body.shape_category == "small"
                        and sum(shape.shape.body.hold) == False
                    ):
                        return shape.shape

                    elif shape.shape.body.shape_category != "small":
                        if (self.map_name == 2005 or self.map_name == 2006) and False:
                            if are_perpendicular(
                                robot.body.angle, shape.shape.body.angle
                            ):
                                return shape.shape
                        else:
                            return shape.shape
            except:
                continue
        # if shapes:
        #     return shapes[0].shape
        return None

    def set_body_unmovable(self, body):
        body.body_type = pm.Body.STATIC

    def set_body_movable(self, body):
        body.body_type = pm.Body.DYNAMIC
        if body.shape_category == "small":
            body.mass = body.original_mass
            body.moment = 0.001
        else:
            body.mass = body.original_mass
            body.moment = 1
            body.damping = 0.1
        body.friction = 0.5
        # body.velocity = (0, 0)
        # body.angular_velocity = 0

    def set_body_part_movable(self, body):
        body.body_type = pm.Body.DYNAMIC
        body.mass = self.slowly_movable_mass
        body.moment = 0.1
        # body.damping = 0
        # body.friction = 0

    def step(self, actions) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # def debug_forces_and_velocity(body):
        #     print(f"Velocity: {body.velocity}, Angular Velocity: {body.angular_velocity}")
        #     print(f"Total Force: {body.force}, Total Torque: {body.torque}")

        # for body_ in self._space.bodies:
        #     if(body_.body_type == pm.Body.DYNAMIC):
        #         debug_forces_and_velocity(body_)

        if True:
            # self._robots[0].body.velocity = (0.0, 0.0)
            # self._robots[0].body.angular_velocity = 0
            # self._robots[1].body.velocity = (0.0, 0.0)
            # self._robots[1].body.angular_velocity = 0
            for i in range(2):
                if (
                    self.if_hold[i]
                    and self._robots[i].holded_item.body.shape_category == "small"
                ):
                    self._robots[i].holded_item.body.velocity = (0.0, 0.0)
                    self._robots[i].holded_item.body.angular_velocity = 0
                if (
                    self.if_hold[i]
                    and self._robots[i].holded_item.body.shape_category == "middle"
                ):
                    self._robots[i].holded_item.body.velocity = (0.0, 0.0)
                    self._robots[i].holded_item.body.angular_velocity = 0
                if (
                    self.if_hold[i]
                    and self._robots[i].holded_item.body.shape_category == "large"
                ):
                    self._robots[i].holded_item.body.velocity = (0.0, 0.0)
                    self._robots[i].holded_item.body.angular_velocity = 0
            # if(self.if_hold[0] and abs(actions[0][0]) + abs(actions[0][1]) == 0):
            #     clipped = np.clip(self.shape_in_front[0].body.velocity, -0.001, 0.001)
            #     self.shape_in_front[0].body.velocity = (clipped[0], clipped[1])
            #     self.shape_in_front[0].body.angular_velocity = np.clip(self.shape_in_front[0].body.angular_velocity, -0.001, 0.001)
            # if(self.if_hold[1] and abs(actions[1][0]) + abs(actions[1][1]) == 0):
            #     clipped = np.clip(self.shape_in_front[1].body.velocity, -0.001, 0.001)
            #     self.shape_in_front[1].body.velocity = (clipped[0], clipped[1])
            #     self.shape_in_front[1].body.angular_velocity = np.clip(self.shape_in_front[1].body.angular_velocity, -0.001, 0.001)
        for i, action in enumerate(actions):
            if abs(action[0]) == 0:
                self._robots[i].body.velocity = (0.0, 0.0)
            if self._robots[i].holded_item is not None:
                if self._robots[i].holded_item.body.mass == self.slowly_movable_mass:
                    self._robots[i].slow_movement = True
                if self._robots[i].holded_item.body.shape_category == "small":
                    self._robots[i].small_items_angle = True
                else:
                    self._robots[i].small_items_angle = False

            if (
                self._robots[i].holded_item is not None
                and self._robots[1 - i].holded_item is not None
            ):
                if (
                    self._robots[i].holded_item.body.id
                    == self._robots[1 - i].holded_item.body.id
                ):
                    self._robots[i].move_together = True
                    self._robots[1 - i].move_together = True
                else:
                    self._robots[i].move_together = False
                    self._robots[1 - i].move_together = False
            else:
                self._robots[i].move_together = False
                self._robots[1 - i].move_together = False

            self._robots[i].set_action(actions[i][0:2])
            if_hold = actions[i][2]

            if if_hold:
                if self.hold_joints[i] is not None:
                    self.realease_front(self._robots[i], i)
                    self.if_hold[i] = False

                else:
                    shape_in_front = self.check_for_object_in_front(
                        self._space, self._robots[i]
                    )
                    if shape_in_front is not None:
                        self.grab_front(shape_in_front, self._robots[i], i)

                        self.if_hold[i] = True

            # self.apply_friction(self._space, 100)

            self._phys_steps_on_frame()
            self._episode_steps += 1

            reward = self.get_reward()

            terminated = False
            truncated = False
            eval_score = self.global_score()
            if eval_score == 1.0:
                terminated = True
            info = {}
            if self.max_episode_steps is not None:
                if self._episode_steps >= self.max_episode_steps:
                    info["TimeLimit.truncated"] = True
                    truncated = True
            if terminated or truncated:
                eval_score = self.global_score()
                assert (
                    0 <= eval_score <= 1
                ), f"eval score {eval_score} out of range for env {self}"
            info.update(eval_score=eval_score)
        obs = self.get_encoded_state()
        return obs, reward, terminated, truncated, info

    def render(self, mode="rgb_array") -> Optional[np.ndarray]:
        for ent in self._entities:
            ent.pre_draw()

        self._renderer_func()

        obs = self.renderer.render()
        if mode == "human":
            if self.viewer is None:
                pygame.init()
                self.viewer = pygame.display.set_mode((self.res_hw[1], self.res_hw[0]))
                pygame.display.set_caption("Moving Out Environment")

            obs = obs.transpose(1, 0, 2)

            pygame.surfarray.blit_array(self.viewer, obs)
            pygame.display.flip()
            # print(pygame.event.get())
        elif mode == "rgb_array":
            return obs

    def close(self) -> None:
        if self.renderer:
            self.renderer.close()
            self.renderer = None
        if self.viewer:
            self.viewer.close()
            self.viewer = None
