import copy
import math
import os
from typing import Any, Dict, Tuple

import moving_out.entities as en
import moving_out.render as magical_rd
import numpy as np
import pygame
import pymunk as pm
import yaml
from moving_out.base_env import BaseEnv
from moving_out.env_parameters import MOVINGOUT_PATH, CACHE_DISTANCE
from moving_out.planner import discrete_env
from moving_out.planner.bfs import bfs
from moving_out.utils.states_encoding import StatesEncoder


class MovingOutEnv(BaseEnv):
    def __init__(
        self,
        use_state: bool = True,
        map_name=None,
        dense_rewards_setting=None,
        add_noise_to_item=False,
        max_episode_steps=1000,
        reward_config_name="default_rewards",
        reward_setting: str = "sparse",
    ) -> None:
        """Constructor.

        Args:
            use_state: Whether to use states rather than pixels for the
                observation space.
            use_dense_reward: Whether to use a dense reward or a sparse one.
            rand_layout_full: Whether to randomize the poses of the items.
            rand_shapes: Whether to randomize the shapes of the items.
            rand_colors: Whether to randomize the colors of the items and the
                goal zone.
        """
        self.states_convertor = StatesEncoder()
        super().__init__(max_episode_steps=max_episode_steps)
        self.reward_setting = reward_setting
        self.use_state = use_state
        self.dense_rewards_setting = dense_rewards_setting

        self.add_noise_to_item = add_noise_to_item

        self.init_rewards(reward_config_name, reward_setting)

        self.map_name = map_name
        self.if_cache_founded = False
        self.loaded_map_id = None
        if map_name is not None:
            self.reset(self.map_name)

    def update_last_state(self, state):
        self.last_state = copy.deepcopy(state)
        self.last_overlapped_items = list(self.get_overlapped_items())

    def get_all_states(self):
        return {
            "robot_1": {
                "pos": list(self._robots[0].body.position),
                "angle": self._robots[0].body.angle,
                "hold": self._robots[0].hold,
                "velocity": list(self._robots[0].body.velocity),
                "angular_velocity": self._robots[0].body.angular_velocity,
                "holded_item_id": (
                    self._robots[0].holded_item.body.custom_id
                    if self._robots[0].holded_item != None
                    else None
                ),
                "holded_item_pos": (
                    list(self._robots[0].holded_item.body.position)
                    if self._robots[0].holded_item != None
                    else None
                ),
                "holded_item_category": (
                    self._robots[0].holded_item.body.shape_category
                    if self._robots[0].holded_item != None
                    else None
                ),
            },
            "robot_2": {
                "pos": list(self._robots[1].body.position),
                "angle": self._robots[1].body.angle,
                "hold": self._robots[1].hold,
                "velocity": list(self._robots[1].body.velocity),
                "angular_velocity": self._robots[1].body.angular_velocity,
                "holded_item_id": (
                    self._robots[1].holded_item.body.custom_id
                    if self._robots[1].holded_item != None
                    else None
                ),
                "holded_item_pos": (
                    list(self._robots[1].holded_item.body.position)
                    if self._robots[1].holded_item != None
                    else None
                ),
                "holded_item_category": (
                    self._robots[1].holded_item.body.shape_category
                    if self._robots[1].holded_item != None
                    else None
                ),
            },
            "items": [
                {
                    "pos": list(shape.shape_body.position),
                    "angle": shape.shape_body.angle,
                    "shape": shape.shape_type,
                    "size": shape.shape_category,
                    "shape_scale": shape.shape_size,
                    "color": shape.color,
                    "id": shape.object_id,
                    "hold":shape.shape_body.hold
                }
                for shape in self.__items_shapes
            ],
            "target_area": self.target_areas,
            "target_color": self.target_color,
            "walls": self.walls,
        }

    def begin(self, arbiter, space, data):
        # Get the contact points
        contact_points = arbiter.contact_point_set.points
        for contact in contact_points:
            position = contact.point_a  # The contact point on the first shape
            distance = contact.distance  # Penetration distance
            # normal = contact.normal  # Collision normal
            # print(f"Contact Point: {position}, Distance: {distance}")
        return True

    def pre_solve(self, arbiter, space, data):
        # Get the contact points before the collision response
        contact_points = arbiter.contact_point_set.points
        for contact in contact_points:
            position = contact.point_a  # The contact point on the first shape
            distance = contact.distance  # Penetration distance
            # normal = contact.normal  # Collision normal
            # print(f"[pre_solve] Contact Point: {position}, Distance: {distance}")

        # Return True to process the collision, False to ignore it
        return True

    def post_solve(self, arbiter, space, data):
        # This is after the collision has been processed
        total_impulse = arbiter.total_impulse
        # print(f"[post_solve] Total impulse: {total_impulse}")
        # for shape in arbiter.shapes:
        #     body = shape.body
        #     body.velocity = body.velocity * 0.1
        #     body.angular_velocity = body.angular_velocity * 0.1
        # You can also retrieve the contact points here if needed
        contact_points = arbiter.contact_point_set.points
        for contact in contact_points:
            position = contact.point_a
            # print(f"[post_solve] Contact Point after collision: {position}")

    def init_discrete_map(self, walls):
        walls = [x for x in walls if x[0] != None]
        if len(walls) == 0:
            self.dicrete_map = None
            self.distance_caluclation = "euclidean"
        else:
            X_dimensions = np.array([(-1.2, -1.2), (1.2, 1.2)])
            Obstacles = np.array(walls)
            self.grid_resolution = 0.05

            grid = discrete_env.discretize_environment(
                X_dimensions, Obstacles, self.target_areas, self.grid_resolution
            )
            self.dicrete_map = grid
            self.distance_caluclation = "astar"
        return None

    def update_env_by_given_state(self, states_dict, reset = False):
        # self.reset(self.map_name)
        robot_1_pos = states_dict["states"]["robot_1"]["pos"]
        robot_1_angle = states_dict["states"]["robot_1"]["angle"]
        robot_1_hold = states_dict["states"]["robot_1"]["hold"]

        robot_2_pos = states_dict["states"]["robot_2"]["pos"]
        robot_2_angle = states_dict["states"]["robot_2"]["angle"]
        robot_2_hold = states_dict["states"]["robot_2"]["hold"]
        if "target_color" not in states_dict["states"]:
            states_dict["states"]["target_color"] = self.target_color

        def format_objects(objects):
            formated_objects = {}
            for obj in objects:
                formated_objects[str(obj["id"])] = obj
                obj["mass"] = 1 # Not the real mass, just for the reset
            return formated_objects
        
        if reset:
            # Delete old entities/space.
            self._entities = []
            self._space = None
            self._robots = []

            self.if_hold = [False, False]
            self.hold_joints = [None, None]
            self.motor_joints = [None, None]
            self.two_robot_joints = None
            self.shape_in_front = [None, None]

            self.renderer.reset_geoms()

            self._space = pm.Space()
            self._space.collision_slop = 0.001
            self._space.iterations = self.phys_iter

            # Set up robot and arena.
            arena_l, arena_r, arena_b, arena_t = self.ARENA_BOUNDS_LRBT
            self._arena = en.ArenaBoundaries(
                left=arena_l, right=arena_r, bottom=arena_b, top=arena_t
            )
            self._arena_w = arena_r - arena_l
            self._arena_h = arena_t - arena_b
            self.add_entities([self._arena])

            formated_objects = format_objects(states_dict["states"]["items"])
            self.on_reset(
                robot_1_pos,
                robot_1_angle,
                robot_2_pos,
                robot_2_angle,
                states_dict["states"]["walls"],
                objects = formated_objects,
                target_areas = states_dict["states"]["target_area"],
                target_color = states_dict["states"]["target_color"],
            )

        self._robots[0].body.position = robot_1_pos
        self._robots[0].body.angle = robot_1_angle
        self._robots[0].body.velocity = states_dict["states"]["robot_1"]["velocity"]
        self._robots[0].body.angular_velocity = states_dict["states"]["robot_1"][
            "angular_velocity"
        ]
        self._robots[0].hold = states_dict["states"]["robot_1"]["hold"]
        self._robots[0].body.velocity = states_dict["states"]["robot_1"]["velocity"]
        self._robots[0].body.angular_velocity = states_dict["states"]["robot_1"][
            "angular_velocity"
        ]

        self._robots[1].body.position = robot_2_pos
        self._robots[1].body.angle = robot_2_angle
        self._robots[1].body.velocity = states_dict["states"]["robot_2"]["velocity"]
        self._robots[1].body.angular_velocity = states_dict["states"]["robot_2"][
            "angular_velocity"
        ]
        self._robots[1].hold = states_dict["states"]["robot_2"]["hold"]
        self._robots[1].body.velocity = states_dict["states"]["robot_2"]["velocity"]
        self._robots[1].body.angular_velocity = states_dict["states"]["robot_2"][
            "angular_velocity"
        ]

        for i, shape in enumerate(self.__items_shapes):
            # self.__items_shapes[i].shape_body.position = [0, 0]
            self.__items_shapes[i].shape_body.position = states_dict["states"][
                "items"
            ][i]["pos"]
            self.__items_shapes[i].shape_body.angle = states_dict["states"]["items"][
                i
            ]["angle"]
            self.__items_shapes[i].shape_body.hold = [False, False]

            self.__items_shapes[i].shape_body.body_type = pm.Body.DYNAMIC

        if robot_1_hold:
            shape_in_front = self.check_for_object_in_front(
                self._space, self._robots[0]
            )
            if shape_in_front is not None:
                self.grab_front(shape_in_front, self._robots[0], 0)

        if robot_2_hold:
            shape_in_front = self.check_for_object_in_front(
                self._space, self._robots[1]
            )
            if shape_in_front is not None:
                self.grab_front(shape_in_front, self._robots[1], 1)

    def on_reset(
        self,
        robot_1_pos,
        robot_1_angle,
        robot_2_pos,
        robot_2_angle,
        walls,
        objects,
        target_areas,
        target_color,
    ) -> None:
        self.target_areas = target_areas
        self.walls = walls

        robot_1 = self._make_robot(robot_1_pos, robot_1_angle, id=1)
        robot_2 = self._make_robot(robot_2_pos, robot_2_angle, id=2)
        self.target_color = target_color

        self.__sensor_ref = []
        for ta in self.target_areas:
            sensor = en.GoalRegion(
                *ta,
                self.target_color,
                dashed=False,
            )
            self.add_entities([sensor])
            self.__sensor_ref.append(sensor)

        valid_shapes = []
        for i in range(len(objects)):
            if str(i) in objects.keys():
                if objects[str(i)] == None:
                    continue
                objects[str(i)]["id"] = i
                valid_shapes.append(objects[str(i)])
        self.__items_shapes = [
            self._make_shape(
                shape_type=_shape["shape"],
                color_name=_shape["color"],
                init_pos=(_shape["pos"][0], _shape["pos"][1]),
                init_angle=_shape["angle"],
                shape_size=_shape["shape_scale"],
                shape_category=_shape["size"],
                pickable=True,
                objects_id=_shape["id"],
                mass=_shape["mass"],
            )
            for _shape in valid_shapes
        ]

        self.add_entities(self.__items_shapes)
        # Add robot last for draw order reasons.
        self.add_entities([robot_1, robot_2])

        self.add_walls(walls)
        self.init_discrete_map(walls)
        robot_1.shape.collision_type = 1
        robot_2.shape.collision_type = 1

        for shape in self.__items_shapes:
            shape.shapes[0].collision_type = 2

        handler = self._space.add_collision_handler(2, 2)
        handler.begin = self.begin
        handler.pre_solve = self.pre_solve
        handler.post_solve = self.post_solve

        # Block lookup index.
        self.__ent_index = en.EntityIndex(self.__items_shapes)

    def add_walls(self, walls):
        for wall in walls:
            if wall[0] == None:
                continue
            lines_list = [
                [wall[0], (wall[1][0], wall[0][1])],
                [wall[0], (wall[0][0], wall[1][1])],
                [wall[1], (wall[1][0], wall[0][1])],
                [wall[1], (wall[0][0], wall[1][1])],
            ]
            # print(lines_list)
            for line in lines_list:
                segment = pm.Segment(self._space.static_body, line[0], line[1], 0.01)
                segment.friction = 0.8
                self._space.add(segment)

                # poly = magical_rd.Poly([line[0], line[1]], outline=True)
                # poly = magical_rd.Poly([line[0], line[1]], outline=True)
                # poly.color = (1, 1, 1)
                # self.renderer.add_geom(poly)
            poly = magical_rd.make_rect(
                wall[1][0] - wall[0][0],
                wall[1][1] - wall[0][1],
                outline=True,
                dashed=False,
            )
            poly.color = (0, 0, 0)
            poly.outline_color = (0, 0, 0)
            goal_xform = magical_rd.Transform()
            goal_xform.reset(
                translation=(np.array(wall[0]) + np.array(wall[1])) / 2.0, rotation=0
            )
            poly.add_transform(goal_xform)
            self.renderer.add_geom(poly)

    def get_overlapped_items(self):
        overlap_ents = set()
        for i in range(len(self.__sensor_ref)):
            overlap_ents = overlap_ents | self.__sensor_ref[i].get_overlapping_ents(
                contained=True, ent_index=self.__ent_index
            )
        return overlap_ents

    def global_score(self) -> float:
        # score = number of items entirely contained in goal zone / 3
        overlap_ents = self.get_overlapped_items()
        target_set = set(self.__items_shapes)
        n_overlap_targets = len(target_set & overlap_ents)
        if n_overlap_targets != 0:
            _ = 1 + 1
        score = n_overlap_targets / len(target_set)
        if len(overlap_ents) == 0:
            score = 0
        return score

    def sparse_rewards_for_one_step(self):
        # if(not self.use_dense_reward):
        #     return [0, 0]
        current_state = self.get_all_states()

        last_state = self.last_state
        last_overlapped_items = list(self.last_overlapped_items)
        overlapped_items = list(self.get_overlapped_items())

        last_overlapped_items_ids = ids = [x.object_id for x in last_overlapped_items]
        overlapped_items_ids = [x.object_id for x in overlapped_items]

        rewards = [0, 0]

        robots = ["robot_1", "robot_2"]

        for i, robot in enumerate(robots):
            if (
                current_state[robot]["holded_item_id"] != None
                and last_state[robot]["holded_item_id"] != None
            ):
                if (
                    last_state[robot]["holded_item_id"] in last_overlapped_items_ids
                    and current_state[robot]["holded_item_id"]
                    not in overlapped_items_ids
                ):
                    if last_state[robot]["holded_item_category"] == "small":
                        rew = self.rewards_config["small_items"][
                            "move_item_out_of_target_areas"
                        ]
                        rew_the_other = self.rewards_config["small_items"][
                            "move_item_out_of_target_areas_rewards_to_the_other"
                        ]
                    elif last_state[robot]["holded_item_category"] == "middle":
                        rew = self.rewards_config["middle_and_large"][
                            "move_item_out_of_target_areas"
                        ]
                        rew_the_other = 0
                    elif last_state[robot]["holded_item_category"] == "large":
                        rew = self.rewards_config["middle_and_large"][
                            "move_item_out_of_target_areas"
                        ]
                        rew_the_other = 0
                    else:
                        print("Invalid shape category")
                        exit(0)
                    rewards[i] += rew
                    rewards[1 - i] += rew_the_other

                if (
                    last_state[robot]["holded_item_id"] not in last_overlapped_items_ids
                    and current_state[robot]["holded_item_id"] in overlapped_items_ids
                ):
                    if last_state[robot]["holded_item_category"] == "small":
                        rew = self.rewards_config["small_items"][
                            "move_items_to_target_areas"
                        ]
                        rew_the_other = self.rewards_config["small_items"][
                            "move_items_to_target_areas_rewards_to_the_other"
                        ]
                    elif last_state[robot]["holded_item_category"] == "middle":
                        rew = self.rewards_config["middle_and_large"][
                            "move_items_to_target_areas"
                        ]
                        rew_the_other = 0
                        # rew_the_other = self.rewards_config["middle_items"]["move_items_to_target_areas_rewards_to_the_other"]
                    elif last_state[robot]["holded_item_category"] == "large":
                        rew = self.rewards_config["middle_and_large"][
                            "move_items_to_target_areas"
                        ]
                        rew_the_other = 0
                        # rew_the_other = self.rewards_config["large_items"]["move_items_to_target_areas_rewards_to_the_other"]
                    else:
                        print("Invalid shape category")
                        exit(0)

                    rewards[i] += rew
                    rewards[1 - i] += rew_the_other

        return rewards

    def get_overlapped_items_by_category(self):
        self.global_score()
        shape_category = {"small": [0, 0], "middle": [0, 0], "large": [0, 0]}
        for i in list(self.get_overlapped_items()):
            shape_category[i.shape_category][0] += 1
        for i in list(self.__items_shapes):
            shape_category[i.shape_category][1] += 1
        return shape_category

    def waypoints(self):
        waypoints = {1000: {"noholding": [[-1.2, 1.2], []]}}

    def load_cache_distance(self, map_name=None):
        current_file_path = os.path.abspath(__file__)
        from pathlib import Path

        mapping = {
            1000: Path("..") / "maps" / "maps_v1" / "1000_ROOM.npy",
            1001: Path("..") / "maps" / "maps_v1" / "1001_ROOM.npy",
            1002: Path("..") / "maps" / "maps_v1" / "1002_ROOM.npy",
            1003: Path("..") / "maps" / "maps_v1" / "1003_ROOM.npy",
            2000: Path("..") / "maps" / "maps_v1" / "2000_ROOM.npy",
            2001: Path("..") / "maps" / "maps_v1" / "2001_ROOM.npy",
            2002: Path("..") / "maps" / "maps_v1" / "2002_ROOM.npy",
            2003: Path("..") / "maps" / "maps_v1" / "2003_ROOM.npy",
            2004: Path("..") / "maps" / "maps_v1" / "2004_ROOM.npy",
            2005: Path("..") / "maps" / "maps_v1" / "2005_ROOM.npy",
            2006: Path("..") / "maps" / "maps_v1" / "2006_ROOM.npy",
            2007: Path("..") / "maps" / "maps_v1" / "2007_ROOM.npy",
        }
        cache_path = mapping[int(map_name)]
        cache_path = os.path.join(os.path.dirname(current_file_path), cache_path)
        cache_exist = os.path.exists(cache_path)
        if int(map_name) not in mapping.keys() or not cache_exist:
            print("------------------------------------------------")
            print(
                "WARNING: No cache founded for this map, the runing will be very slow!!!!!!!"
            )
            print("------------------------------------------------")
            self.distance_cache = None
            return False
        else:
            self.loaded_map_id = map_name
            # print("INFO: Cache founded for this map, the running will be faster")
            cache_path = mapping[int(map_name)]
            cache_path = os.path.join(os.path.dirname(current_file_path), cache_path)
            self.distance_cache = np.load(cache_path, allow_pickle=True)
            self.distance_cache = sorted(
                self.distance_cache, key=lambda x: (x[0][0], x[0][1], x[1][0], x[1][1])
            )

            distance_to_target_areas_cache_path = cache_path.replace(
                ".json", "_to_goal_region.npy"
            )
            self.distance_to_target_areas_cache = np.load(
                distance_to_target_areas_cache_path, allow_pickle=True
            )
            self.distance_to_target_areas_cache = sorted(
                self.distance_to_target_areas_cache,
                key=lambda x: (x[0][0], x[0][1], x[1][0], x[1][1]),
            )

            self.if_cache_founded = True
            return True

    def get_distance_from_A_to_B_by_A_star(self, A, B):
        start_point = tuple(A)
        goal_point = tuple(B)
        if self.distance_caluclation == "euclidean":
            distance = np.linalg.norm(np.array(start_point) - np.array(goal_point))
            return distance
        cached = self.if_cache_founded
        if cached:
            if start_point[0] > goal_point[0]:
                start_point, goal_point = goal_point, start_point
            x0, y0 = start_point
            x1, y1 = goal_point

            resolution = 0.05
            grid_size_x = 48
            grid_size_y = 48

            i0 = round((x0 + 1.2) / resolution)
            j0 = round((y0 + 1.2) / resolution)
            i1 = round((x1 + 1.2) / resolution)
            j1 = round((y1 + 1.2) / resolution)

            start_index = i0 * grid_size_x + j0
            end_index = i1 * grid_size_y + j1
            index = (start_index) * (grid_size_x * grid_size_y) + end_index

            distance = float(list(self.distance_cache[index])[2]) * resolution
            if distance == 0:
                return None
        else:
            path = bfs(
                self.dicrete_map,
                start_point,
                goal_point,
                goal_is_point=True,
                discrete_start_goal=True,
            )
            if path is None or path is False:
                return None
            distance = len(path) * 0.05
        return distance

    def get_distance_to_closest_item(
        self,
        agent_position,
        items_positions,
        distance_caluclation="euclidean",
        return_item_n_distance=False,
        robot_id=0,
    ):
        if self.map_name == 2005 or self.map_name == 2006:
            items_positions = self.get_all_states()["items"]
            two_pos = items_positions[0]["pos"]
            angle = items_positions[0]["angle"]

            pos_1 = [
                two_pos[0] + math.cos(angle) * 0.2,
                two_pos[1] + math.sin(angle) * 0.1,
            ]
            pos_2 = [
                two_pos[0] - math.cos(angle) * 0.2,
                two_pos[1] - math.sin(angle) * 0.1,
            ]

            if robot_id == 0:
                closest_distance = np.linalg.norm(
                    np.array(agent_position) - np.array(pos_1)
                )
            else:
                closest_distance = np.linalg.norm(
                    np.array(agent_position) - np.array(pos_2)
                )
            return closest_distance
        else:
            position_list = [list(item["pos"]) for item in items_positions]
            closest_distance = 1e5
            index = 0
            for i, pos in enumerate(position_list):
                if distance_caluclation == "astar":
                    # Using A* algorithm to find the shortest path
                    start_point = tuple(agent_position)
                    goal_point = tuple(pos)
                    distance = self.get_distance_from_A_to_B_by_A_star(
                        start_point, goal_point
                    )
                    if distance is None:
                        distance = 1e6
                elif distance_caluclation == "euclidean":
                    distance = np.linalg.norm(np.array(agent_position) - np.array(pos))
                else:
                    print(
                        "Invalid distance calculation method : ", distance_caluclation
                    )
                    exit(0)
                if distance < closest_distance:
                    index = i
                    closest_distance = distance

            if closest_distance == 1e5:
                return None
            if return_item_n_distance:
                return closest_distance, items_positions[i]
            else:
                return closest_distance

    def _get_distance_to_one_target_areas(self, position, target_areas):
        def point_to_rectangle_distance(position, target_areas):
            x, y = position
            
            left = target_areas[0][0]
            right = target_areas[1][0]
            top = target_areas[0][1]
            bottom = target_areas[1][1]

            # Calculate the horizontal and vertical distance from the point to the rectangle
            dx = max(left - x, 0, x - right)  # Shortest distance in x direction
            dy = max(bottom - y, 0, y - top)  # Shortest distance in y direction

            # If the point is inside the rectangle, the distance is 0
            if dx == 0 and dy == 0:
                return 0.0

            # Calculate Euclidean distance
            return np.sqrt(dx**2 + dy**2)

        start_point = tuple(position)
        # # goal_point = tuple(optimal_position)
        # goal_point = [0, 0]
        # for i, item in enumerate(optimal_position):
        #     if item < -1.19:
        #         item = -1.15
        #     if item > 1.19:
        #         item = 1.15
        #     goal_point[i] = item
        # goal_point = tuple(goal_point)

        distance_caluclation = self.distance_caluclation
        if distance_caluclation == "astar":
            distance = (
                len(
                    bfs(
                        self.dicrete_map,
                        start_point,
                        [target_areas],
                        goal_is_point=False,
                        discrete_start_goal=True,
                    )
                )
                * self.grid_resolution
            )
        elif distance_caluclation == "euclidean":
            distance = point_to_rectangle_distance(position, target_areas)
        return distance

    def get_distance_to_target_areas(self, position):
        if isinstance(self.target_areas[0], float):
            target_areas = [self.target_areas]
        else:
            target_areas = self.target_areas

        if self.if_cache_founded and False:
            pass
        else:
            distance_list = []
            for ta in target_areas:
                dis = self._get_distance_to_one_target_areas(position, ta)
                distance_list.append(dis)
            try:
                minimal_distance = min(distance_list)
            except:
                print("GET Wrong Distance", distance_list, "-", position, "-", ta)
                return 1.44
            if minimal_distance is None:
                print("minimal_distance is None", distance_list, "-", position, "-", ta)
                minimal_distance = 1.44
            return minimal_distance

    def init_rewards(self, reward_config_name, reward_setting):
        config_path = os.path.join(MOVINGOUT_PATH, "conf", reward_config_name + ".yaml")
        with open(config_path, "r") as f:
            self.rewards_config = yaml.load(f, Loader=yaml.FullLoader)[reward_setting]

    def get_item_by_id(self, id):
        for item in self.__items_shapes:
            if item.object_id == id:
                return item
        return None

    def get_distance_of_all_items_to_target_areas(self):
        states = self.get_all_states()
        all_items_pos = [x["pos"] for x in states["items"]]
        dis_sum = 0
        for pos in all_items_pos:
            dis_sum += self.get_distance_to_target_areas(pos)
        return dis_sum

    def _init_global_dense_reward(self):
        self.init_distance_to_target_areas = (
            self.get_distance_of_all_items_to_target_areas()
        )

    def global_dense_reward(self):
        distance = self.get_distance_of_all_items_to_target_areas()
        return 1 - distance / self.init_distance_to_target_areas

    def _dense_reward(self) -> float:
        """Mean distance of all items entitity positions to goal zone."""

        if self.reward_setting == "sparse":
            print("Reward Setting is Sparse, no dense reward")
            return 0

        last_state = self.last_state
        last_overlapped_items = self.last_overlapped_items

        current_state = self.get_all_states()

        rewards = [0, 0]

        def rewards_for_unholded_agents_to_closest_items(
            current_state, last_state, i, shape="small"
        ):
            radius = 0.05

            current_angle = current_state[agent]["angle"]
            current_front_pos = np.array(
                current_state[agent]["pos"]
            ) + radius * np.array([-math.sin(current_angle), math.cos(current_angle)])

            last_angle = last_state[agent]["angle"]
            last_front_pos = np.array(last_state[agent]["pos"]) + radius * np.array(
                [-math.sin(last_angle), math.cos(last_angle)]
            )
            # print(np.array(current_state[agent]["pos"]), "--" ,last_front_pos)
            current_items = [
                x for x in list(current_state["items"]) if (((not x["hold"][1 - i])))
            ]
            last_items = [
                x for x in list(last_state["items"]) if (((not x["hold"][1 - i])))
            ]

            current_items_ids = [x["id"] for x in current_items]
            last_items_ids = [x["id"] for x in last_items]

            valid_id = []
            for id in current_items_ids:
                if id in last_items_ids:
                    valid_id.append(id)

            overlapped_items = self.get_overlapped_items()
            overlapped_items_ids = [x.object_id for x in overlapped_items]

            filted_current_items = [
                x
                for x in current_items
                if x["id"] in valid_id and x["id"] not in overlapped_items_ids
            ]
            filted_last_items = [
                x
                for x in last_items
                if x["id"] in valid_id and x["id"] not in overlapped_items_ids
            ]

            current_cloest_distance = self.get_distance_to_closest_item(
                current_front_pos,
                filted_current_items,
                self.distance_caluclation,
                robot_id=i,
            )
            last_cloest_distance = self.get_distance_to_closest_item(
                last_front_pos,
                filted_last_items,
                self.distance_caluclation,
                robot_id=i,
            )
            if current_cloest_distance == None or last_cloest_distance == None:
                rewards = 0
            else:
                rewards = (
                    -(current_cloest_distance - last_cloest_distance)
                    * self.rewards_config["small_items"][
                        "scale_for_agents_get_closer_to_cloest_small_items"
                    ]
                )
            return rewards

        agents = ["robot_1", "robot_2"]
        rewards = [0, 0]
        for i, agent in enumerate(agents):
            if (
                current_state[agent]["holded_item_id"] != None
                and last_state[agent]["holded_item_id"] != None
            ):
                if current_state[agent]["holded_item_category"] == "small":
                    current_distance = self.get_distance_to_target_areas(
                        current_state[agent]["holded_item_pos"]
                    )
                    last_distance = self.get_distance_to_target_areas(
                        last_state[agent]["holded_item_pos"]
                    )

                    if last_distance == None or current_distance == None:
                        rewards[i] += 0
                    else:
                        rewards[i] += (
                            -(current_distance - last_distance)
                            * self.rewards_config["small_items"][
                                "scale_for_agents_get_closer_to_cloest_small_items"
                            ]
                        )

                elif (
                    current_state[agent]["holded_item_category"] == "middle"
                    or current_state[agent]["holded_item_category"] == "large"
                ):
                    current_distance = self.get_distance_to_target_areas(
                        current_state[agent]["holded_item_pos"]
                    )
                    last_distance = self.get_distance_to_target_areas(
                        last_state[agent]["holded_item_pos"]
                    )
                    if last_distance == None or current_distance == None:
                        rewards[i] += 0
                    else:
                        rewards[i] += (
                            -(current_distance - last_distance)
                            * self.rewards_config["middle_and_large"][
                                "scale_for_agents_get_closer_to_target_middle_large_or_target_area"
                            ]
                        )
                # rewards[i] = 0 if rewards[i] < 0.00001 else rewards[i]
            elif (
                current_state[agent]["holded_item_id"] == None
                and last_state[agent]["holded_item_id"] == None
            ):
                if (
                    current_state[agents[1 - i]]["holded_item_category"] == "middle"
                    or current_state[agents[1 - i]]["holded_item_category"] == "large"
                ):
                    if self.map_name == 2005 or self.map_name == 2006:
                        rewards[i] += rewards_for_unholded_agents_to_closest_items(
                            current_state, last_state, i
                        )
                    else:
                        current_distance = self.get_distance_from_A_to_B_by_A_star(
                            current_state[agent]["pos"],
                            current_state[agents[1 - i]]["holded_item_pos"],
                        )
                        if last_state[agents[1 - i]]["holded_item_pos"] is not None:
                            last_distance = self.get_distance_from_A_to_B_by_A_star(
                                last_state[agent]["pos"],
                                last_state[agents[1 - i]]["holded_item_pos"],
                            )
                            if last_distance == None or current_distance == None:
                                rewards[i] += 0
                            else:
                                rewards[i] += (
                                    -(current_distance - last_distance)
                                    * self.rewards_config["middle_and_large"][
                                        "scale_for_agents_get_closer_to_target_middle_large_or_target_area"
                                    ]
                                )
                        else:
                            pass
                else:
                    rewards[i] += rewards_for_unholded_agents_to_closest_items(
                        current_state, last_state, i
                    )
                # rewards[i] = 0 if rewards[i] < 0.00001 else rewards[i]
            elif (
                current_state[agent]["holded_item_id"] == None
                and last_state[agent]["holded_item_id"] != None
            ):
                overlapped_items = self.get_overlapped_items()
                overlapped_items_ids = [x.object_id for x in overlapped_items]

                # current_all_items_id_pos = {x["id"]:x["pos"] for x in list(current_state["items"])}
                # last_all_items_id_pos = {x["id"]:x["pos"] for x in list(last_state["items"])}
                if (
                    current_state[agents[1 - i]]["holded_item_category"] == "middle"
                    or current_state[agents[1 - i]]["holded_item_category"] == "large"
                ):
                    current_distance = self.get_distance_from_A_to_B_by_A_star(
                        current_state[agent]["pos"],
                        current_state[agents[1 - i]]["holded_item_pos"],
                    )
                    if last_state[agents[1 - i]]["holded_item_pos"] is not None:
                        last_distance = self.get_distance_from_A_to_B_by_A_star(
                            last_state[agent]["pos"],
                            last_state[agents[1 - i]]["holded_item_pos"],
                        )
                        rewards[i] += (
                            -(current_distance - last_distance)
                            * self.rewards_config["middle_and_large"][
                                "scale_for_agents_get_closer_to_target_middle_large_or_target_area"
                            ]
                        )
                    else:
                        rewards[i] += rewards_for_unholded_agents_to_closest_items(
                            current_state, last_state, i
                        )
                else:
                    rewards[i] += rewards_for_unholded_agents_to_closest_items(
                        current_state, last_state, i
                    )

                if last_state[agent]["holded_item_id"] in overlapped_items_ids:
                    if last_state[agent]["holded_item_category"] == "small":
                        rewards[i] += self.rewards_config["small_items"][
                            "unhold_small_items_in_target_areas"
                        ]
                    elif (
                        last_state[agent]["holded_item_category"] == "middle"
                        or last_state[agent]["holded_item_category"] == "large"
                    ):
                        rewards[i] += self.rewards_config["middle_and_large"][
                            "unhold_large_items_in_target_areas"
                        ]
                    else:
                        print("Invalid shape category")
                        exit(0)
                else:
                    if last_state[agent]["holded_item_category"] == "small":
                        rewards[i] += self.rewards_config["small_items"][
                            "unhold_small_items_which_not_inside_target_area"
                        ]
                    elif (
                        last_state[agent]["holded_item_category"] == "middle"
                        or last_state[agent]["holded_item_category"] == "large"
                    ):
                        if last_state[agent]["holded_item_id"] != current_state[
                            agents[1 - i]
                        ]["holded_item_id"] and (
                            current_state[agents[1 - i]]["holded_item_category"]
                            == "middle"
                            or current_state[agents[1 - i]]["holded_item_category"]
                            == "large"
                        ):
                            pass
                        else:
                            rewards[i] += self.rewards_config["middle_and_large"][
                                "unhold_items_which_not_inside_target_area"
                            ]

            elif (
                current_state[agent]["holded_item_id"] != None
                and last_state[agent]["holded_item_id"] == None
            ):
                current_distance = self.get_distance_to_target_areas(
                    current_state[agent]["holded_item_pos"]
                )
                id = current_state[agent]["holded_item_id"]
                for item in last_state["items"]:
                    if item["id"] == id:
                        last_item_pos = item["pos"]
                        break
                last_distance = self.get_distance_to_target_areas(last_item_pos)
                if last_distance == None or current_distance == None:
                    rewards[i] += 0
                else:
                    rewards[i] += (
                        -(current_distance - last_distance)
                        * self.rewards_config["small_items"][
                            "scale_for_agents_get_closer_to_cloest_small_items"
                        ]
                    )
                overlapped_items = self.get_overlapped_items()
                overlapped_items_ids = [x.object_id for x in overlapped_items]

                rewards[i] -= rewards_for_unholded_agents_to_closest_items(
                    current_state, last_state, i
                )

                if current_state[agent]["holded_item_category"] == "small":
                    if current_state[agent]["holded_item_id"] in overlapped_items_ids:
                        rewards[i] += self.rewards_config["small_items"][
                            "hold_small_items_in_target_areas"
                        ]
                    else:
                        rewards[i] += self.rewards_config["small_items"][
                            "hold_small_items_which_not_inside_target_area"
                        ]
                elif (
                    current_state[agent]["holded_item_category"] == "middle"
                    or current_state[agent]["holded_item_category"] == "large"
                ):
                    if current_state[agent]["holded_item_id"] in overlapped_items_ids:
                        rewards[i] += self.rewards_config["middle_and_large"][
                            "hold_large_items_in_target_areas"
                        ]
                    else:
                        if current_state[agent]["holded_item_id"] != current_state[
                            agents[1 - i]
                        ]["holded_item_id"] and (
                            current_state[agents[1 - i]]["holded_item_category"]
                            == "middle"
                            or current_state[agents[1 - i]]["holded_item_category"]
                            == "large"
                        ):
                            rewards[i] += self.rewards_config["middle_and_large"][
                                "hold_items_but_different_with_the_other"
                            ]
                        else:
                            rewards[i] += self.rewards_config["middle_and_large"][
                                "hold_items_which_not_inside_target_area"
                            ]

        return rewards

    def clone(self, cache_file=None):
        self.renderer.screen = None
        if cache_file is not None:
            self.distance_cache = None
        env = copy.deepcopy(self)
        if cache_file is not None:
            self.distance_cache = cache_file
            env.distance_cache = cache_file
        env._space = env._robots[0].body.space
        env.renderer.screen = pygame.Surface((env.renderer.height, env.renderer.width))
        return env

    def _sparse_reward(self) -> float:
        """Fraction of items entities inside goal zone."""
        return self.sparse_rewards_for_one_step()

    def get_reward(self) -> float:
        step_cost = self.rewards_config["step_cost"]
        if self.reward_setting == "dense":
            reward = [0, 0]
            reward[0] = self._dense_reward()[0] + self._sparse_reward()[0] + step_cost
            reward[1] = self._dense_reward()[1] + self._sparse_reward()[1] + step_cost
            return reward
        elif self.reward_setting == "sparse":
            reward = [0, 0]
            reward[0] = self._sparse_reward()[0] + step_cost
            reward[1] = self._sparse_reward()[1] + step_cost
            return reward
        elif self.reward_setting == "mixed":
            reward = [0, 0]
            reward[0] = (
                self.global_dense_reward() + self._sparse_reward()[0] + step_cost
            )
            reward[1] = (
                self.global_dense_reward() + self._sparse_reward()[1] + step_cost
            )
            return reward

    def reset(self, map_name=None, add_noise_to_item=None) -> np.ndarray:
        if add_noise_to_item is True or self.add_noise_to_item is True:
            add_noise_to_item = True
        obs = super().reset(map_name=map_name, add_noise_to_item=add_noise_to_item)

        self._init_global_dense_reward()
        if (
            map_name == self.loaded_map_id and self.load_cache_distance != None
        ) or self.distance_caluclation == "euclidean":
            pass
        elif CACHE_DISTANCE:
            self.load_cache_distance(map_name)
        else:
            pass
        if self.use_state:
            obs = self.get_state()
            return obs
        return obs

    def get_state(self):
        obs = self.get_all_states()
        obs = self.states_convertor.get_state_by_current_obs_states(obs)
        return obs

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        last_state = self.get_all_states()
        self.update_last_state(last_state)

        obs, rew, terminated, truncated, info = super().step(action)

        info["global_score"] = self.global_score()

        if self.use_state:
            obs = self.get_state()
        return obs, rew, terminated, truncated, info
