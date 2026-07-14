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
        arena_size=[1.2, 1.2],
        cograb_curriculum: float = 0.0,
        cograb_teleport_frac: float = 0.5,
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
        self.cograb_curriculum = float(cograb_curriculum)
        self.cograb_teleport_frac = float(cograb_teleport_frac)
        super().__init__(max_episode_steps=max_episode_steps, arena_size=arena_size)
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
        # custom_id of the item each robot is currently holding (or None).
        # Used to populate the per-item "hold" flag below, which the dense
        # reward and the state encoder both rely on.
        _held_ids = []
        for _r in (0, 1):
            _robot = self._robots[_r]
            if (
                self.if_hold[_r]
                and _robot.holded_item is not None
                and _robot.holded_item.body is not None
            ):
                _held_ids.append(getattr(_robot.holded_item.body, "custom_id", None))
            else:
                _held_ids.append(None)
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
                    "hold": [
                        _held_ids[0] == shape.object_id,
                        _held_ids[1] == shape.object_id,
                    ],
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

    def _pos_in_target_areas(self, pos, margin=0.0):
        # Center-in-rectangle test, with the rectangle shrunk by `margin` on
        # every side. Used for ALL reward judgments instead of the physics
        # sensor's strict full-containment (contained=True), which flickers
        # when an item at the zone edge is grabbed/rotated and made grab/
        # release reward judgments inconsistent (farmable).
        if pos is None:
            return False
        x, y = float(pos[0]), float(pos[1])
        for area in self.target_areas:
            (x1, y1), (x2, y2) = area[0], area[1]
            if (
                min(x1, x2) + margin <= x <= max(x1, x2) - margin
                and min(y1, y2) + margin <= y <= max(y1, y2) - margin
            ):
                return True
        return False

    def _item_circumradius(self, item_id):
        # Rotation-invariant bounding radius of an item, cached per item id.
        # Center inside the margin-shrunk rectangle => shape fully inside the
        # zone, so reward-delivered implies score-delivered (global_score uses
        # full containment). Without this margin the reward fired for items
        # merely STRADDLING the zone edge (center in, body out), letting the
        # policy park items at the boundary: dense reward rose, score stayed 0.
        cache = getattr(self, "_circumradius_cache", None)
        if cache is None:
            cache = self._circumradius_cache = {}
        if item_id in cache:
            return cache[item_id]
        r = 0.05
        for shp in self.__items_shapes:
            if shp.object_id != item_id:
                continue
            radii = []
            for pm_shape in shp.shape_body.shapes:
                if hasattr(pm_shape, "get_vertices"):
                    radii += [v.length for v in pm_shape.get_vertices()]
                elif hasattr(pm_shape, "radius"):
                    radii.append(float(pm_shape.radius))
            if radii:
                r = max(radii)
            break
        cache[item_id] = r
        return r

    def _item_in_target(self, item_id, pos):
        return self._pos_in_target_areas(pos, margin=self._item_circumradius(item_id))

    def _item_ids_in_target(self, state):
        return set(
            it["id"]
            for it in state["items"]
            if self._item_in_target(it["id"], it["pos"])
        )

    def sparse_rewards_for_one_step(self):
        # Reward items ENTERING / LEAVING the target areas based on the items'
        # own in-target transition between steps (center-based, see
        # _pos_in_target_areas), NOT on which robot is holding them and NOT on
        # the jitter-prone full-containment sensor. Every entry gives +20 and
        # every exit gives -20, regardless of holding. Credit is assigned to
        # the robot currently holding the item, else the nearest robot.
        current_state = self.get_all_states()

        last_in_ids = self._item_ids_in_target(self.last_state)
        now_in_ids = self._item_ids_in_target(current_state)

        entered = now_in_ids - last_in_ids
        left = last_in_ids - now_in_ids

        rewards = [0, 0]
        if not entered and not left:
            return rewards

        robots = ["robot_1", "robot_2"]
        id_to_shape = {s.object_id: s for s in self.__items_shapes}

        def responsible_robot(item_id):
            # robot currently holding the item, else the nearest robot
            for r in (0, 1):
                if current_state[robots[r]]["holded_item_id"] == item_id:
                    return r
            shp = id_to_shape.get(item_id)
            if shp is None:
                return 0
            ipos = np.array(shp.shape_body.position)
            dists = [
                np.linalg.norm(np.array(current_state[robots[r]]["pos"]) - ipos)
                for r in (0, 1)
            ]
            return int(np.argmin(dists))

        def reward_pair(item_id, key):
            shp = id_to_shape.get(item_id)
            cat = shp.shape_category if shp is not None else "small"
            if cat == "small":
                cfg = self.rewards_config["small_items"]
                return cfg[key], cfg.get(key + "_rewards_to_the_other", 0)
            return self.rewards_config["middle_and_large"][key], 0

        for item_id in entered:
            i = responsible_robot(item_id)
            rew, rew_the_other = reward_pair(item_id, "move_items_to_target_areas")
            rewards[i] += rew
            rewards[1 - i] += rew_the_other
        for item_id in left:
            i = responsible_robot(item_id)
            rew, rew_the_other = reward_pair(item_id, "move_item_out_of_target_areas")
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
        # Load the flood-fill distance cache generated by
        # scripts/gen_dist_cache.py (<map>.json.distcache.npz). The original
        # implementation only knew integer "ROOM" map ids (and crashed with
        # int("HandOff")), used a pairwise cache whose 48-point linspace grid
        # did not match the runtime 49x49 round((x+1.2)/0.05) indexing, and
        # never set if_cache_founded — so walled maps always fell back to a
        # fresh BFS per distance query (~20 env steps/s).
        from moving_out.env_parameters import AVAILABLE_MAPS, DEFAULT_MAP_PATH

        self.if_cache_founded = False
        self._dist_all_pairs = None
        self._dist_to_goal = None
        json_name = AVAILABLE_MAPS.get(map_name)
        if json_name is None:
            try:
                json_name = AVAILABLE_MAPS.get(int(map_name))
            except (TypeError, ValueError):
                json_name = None
        if json_name is None:
            return False
        cache_path = os.path.join(DEFAULT_MAP_PATH, json_name) + ".distcache.npz"
        if not os.path.exists(cache_path):
            if not getattr(self, "_cache_warned", False):
                self._cache_warned = True
                print("------------------------------------------------")
                print(
                    "WARNING: No distance cache for this map; running will be slow. "
                    f"Generate with scripts/gen_dist_cache.py ({cache_path})"
                )
                print("------------------------------------------------")
            return False
        data = np.load(cache_path)
        self._dist_all_pairs = data["all_pairs"]
        self._dist_to_goal = data["to_goal"]
        self._dist_grid_shape = tuple(int(v) for v in data["shape"])
        self.if_cache_founded = True
        self.loaded_map_id = map_name
        return True

    def _flat_cell(self, pos):
        # NaN-safe: physics blow-ups (large-item joints on rotation maps) can
        # set body positions to NaN; int(round(nan)) raises. Return None so
        # callers fall back gracefully (the old live-BFS path crashed too).
        x, y = float(pos[0]), float(pos[1])
        if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
            return None
        H, W = getattr(self, "_dist_grid_shape", (49, 49))
        i = min(max(int(round((x + 1.2) / 0.05)), 0), H - 1)
        j = min(max(int(round((y + 1.2) / 0.05)), 0), W - 1)
        return i * W + j

    def _unused_legacy_load_cache(self, map_name=None):
        mapping = {}
        cache_path = mapping[int(map_name)]
        if True:
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
        if self.if_cache_founded and self._dist_all_pairs is not None:
            a = self._flat_cell(start_point)
            b = self._flat_cell(goal_point)
            if a is None or b is None:
                return None
            d = self._dist_all_pairs[a, b]
            if d == 65535:
                return None
            return float(d) * 0.05
        cached = False
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

        if (
            self.distance_caluclation == "astar"
            and self.if_cache_founded
            and self._dist_to_goal is not None
        ):
            f = self._flat_cell(position)
            if f is None:
                return 1.44
            d = self._dist_to_goal[f]
            if d == 65535:
                return 1.44
            return float(d) * 0.05
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
        # Merge user-provided overrides (e.g. BenchMARL task.dense_rewards_setting).
        # Previously this constructor argument was stored but NEVER applied, so
        # every override silently fell back to the yaml defaults.
        overrides = getattr(self, "dense_rewards_setting", None)
        if overrides:
            def _deep_merge(dst, src):
                for k, v in dict(src).items():
                    if isinstance(v, dict) or str(type(v)).find("DictConfig") >= 0:
                        node = dst.setdefault(k, {})
                        if isinstance(node, dict):
                            _deep_merge(node, dict(v))
                        else:
                            dst[k] = dict(v)
                    else:
                        dst[k] = v
            try:
                _deep_merge(self.rewards_config, dict(overrides))
            except Exception as e:
                print(f"WARNING: failed to merge dense_rewards_setting: {e}")
        # Canonical unheld shaping: replace all branch-switching approach
        # shaping for unheld agents with ONE potential (min BFS distance to
        # any undelivered item). Kills the reward pump created by toggling
        # between different distance functions.
        self._canon_unheld = bool(
            self.rewards_config.get("middle_and_large", {}).get(
                "canonical_unheld_shaping", False
            )
        )

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
            if getattr(self, "_canon_unheld", False):
                return 0
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
                                    * (0.0 if getattr(self, "_canon_unheld", False) else self.rewards_config["middle_and_large"][
                                        "scale_for_agents_get_closer_to_target_middle_large_or_target_area"
                                    ])
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
                        if current_distance is None or last_distance is None:
                            rewards[i] += 0
                        else:
                            rewards[i] += (
                                -(current_distance - last_distance)
                                * (0.0 if getattr(self, "_canon_unheld", False) else self.rewards_config["middle_and_large"][
                                    "scale_for_agents_get_closer_to_target_middle_large_or_target_area"
                                ])
                            )
                    else:
                        rewards[i] += rewards_for_unholded_agents_to_closest_items(
                            current_state, last_state, i
                        )
                else:
                    rewards[i] += rewards_for_unholded_agents_to_closest_items(
                        current_state, last_state, i
                    )

                # Judge "released in target" by the item's center position
                # (stable), not the flickery containment sensor. The +10 bonus
                # only fires for a GENUINE delivery: the item must have been
                # OUTSIDE the target when this robot grabbed it. Otherwise a
                # grab(-10)/release(+10) cycle inside the zone nets 0 and, with
                # judgment flicker, used to net +15 (farmable).
                released_id = last_state[agent]["holded_item_id"]
                released_pos = None
                for _it in current_state["items"]:
                    if _it["id"] == released_id:
                        released_pos = _it["pos"]
                        break
                released_in_target = self._item_in_target(released_id, released_pos)
                grabbed_in_target = getattr(
                    self, "_grabbed_in_target", [False, False]
                )[i]
                if released_in_target:
                    if not grabbed_in_target:
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
                rewards[i] -= rewards_for_unholded_agents_to_closest_items(
                    current_state, last_state, i
                )

                # Judge "grabbed in target" by the item's center position, and
                # remember it so the release bonus can be limited to genuine
                # deliveries (see the unhold branch above).
                grabbed_now_in_target = self._item_in_target(
                    current_state[agent]["holded_item_id"],
                    current_state[agent]["holded_item_pos"],
                )
                if not hasattr(self, "_grabbed_in_target"):
                    self._grabbed_in_target = [False, False]
                self._grabbed_in_target[i] = grabbed_now_in_target

                if current_state[agent]["holded_item_category"] == "small":
                    if grabbed_now_in_target:
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
                    if grabbed_now_in_target:
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

        # Co-hold potential (opt-in via cohold_potential, default 0): both
        # agents holding the SAME middle/large item is a state potential —
        # +K to both on entering, -K on leaving, so grab/release cycles are
        # net-zero and unfarmable, but the co-held state itself has value.
        cohold_k = self.rewards_config["middle_and_large"].get(
            "cohold_potential", 0.0
        )
        if cohold_k:
            def _coheld(st):
                a, b = agents[0], agents[1]
                return (
                    st[a]["holded_item_id"] is not None
                    and st[a]["holded_item_id"] == st[b]["holded_item_id"]
                    and st[a]["holded_item_category"] in ("middle", "large")
                )
            now_c, was_c = _coheld(current_state), _coheld(last_state)
            if now_c and not was_c:
                rewards[0] += cohold_k
                rewards[1] += cohold_k
            elif was_c and not now_c:
                rewards[0] -= cohold_k
                rewards[1] -= cohold_k

        # Canonical potential for unheld agents: -delta of min BFS distance
        # from the agent's front point to ANY undelivered item, independent
        # of who holds what — a single, never-switching potential.
        if getattr(self, "_canon_unheld", False):
            _ov_ids = {x.object_id for x in self.get_overlapped_items()}
            _scale = self.rewards_config["middle_and_large"][
                "scale_for_agents_get_closer_to_target_middle_large_or_target_area"
            ]

            def _phi(st, agent_key):
                ang = st[agent_key]["angle"]
                front = np.array(st[agent_key]["pos"]) + 0.05 * np.array(
                    [-math.sin(ang), math.cos(ang)]
                )
                best = None
                for it in st["items"]:
                    if it is None or it["id"] in _ov_ids:
                        continue
                    d = self.get_distance_from_A_to_B_by_A_star(
                        list(front), list(it["pos"])
                    )
                    if d is not None and (best is None or d < best):
                        best = d
                return best

            for i, agent in enumerate(agents):
                if current_state[agent]["holded_item_id"] is not None:
                    continue
                cur_d, last_d = _phi(current_state, agent), _phi(last_state, agent)
                if cur_d is not None and last_d is not None:
                    rewards[i] += -(cur_d - last_d) * _scale

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
        # Compute each component ONCE (the old code called _dense_reward()
        # and _sparse_reward() twice per step — pure waste, and expensive on
        # walled maps where they trigger BFS distance queries).
        step_cost = self.rewards_config["step_cost"]
        if self.reward_setting == "dense":
            dense = self._dense_reward()
            sparse = self._sparse_reward()
            return [
                dense[0] + sparse[0] + step_cost,
                dense[1] + sparse[1] + step_cost,
            ]
        elif self.reward_setting == "sparse":
            sparse = self._sparse_reward()
            return [sparse[0] + step_cost, sparse[1] + step_cost]
        elif self.reward_setting == "mixed":
            g = self.global_dense_reward()
            sparse = self._sparse_reward()
            return [g + sparse[0] + step_cost, g + sparse[1] + step_cost]

    def reset(self, map_name=None, add_noise_to_item=None) -> np.ndarray:
        if add_noise_to_item is True or self.add_noise_to_item is True:
            add_noise_to_item = True
        self._grabbed_in_target = [False, False]
        self._circumradius_cache = {}
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
        # Reverse curriculum for two-robot carries: with probability
        # cograb_curriculum, start the episode with one robot already
        # holding a middle/large item and the other posed inside the
        # co-grab window facing it. Set to 0 for standard starts (eval).
        p = float(getattr(self, "cograb_curriculum", 0.0))
        if p > 0.0 and np.random.random() < p:
            self._apply_cograb_curriculum()
        if self.use_state:
            obs = self.get_state()
            return obs
        return obs

    def _apply_cograb_curriculum(self):
        # All-or-nothing: on ANY placement failure, restore every body we
        # touched to its spawn pose — a robot left at a rejected sample
        # (possibly inside a wall) poisons the episode with invalid physics.
        saved = []
        for rid in (0, 1):
            b = self._robots[rid].body
            saved.append((b, tuple(b.position), float(b.angle)))

        def _rollback():
            for b, pos, ang in saved:
                b.position = pos
                b.angle = ang
                b.velocity = (0.0, 0.0)
                b.angular_velocity = 0.0
                try:
                    self._space.reindex_shapes_for_body(b)
                except Exception:
                    pass
            self._space.step(1e-6)

        try:
            states = self.get_all_states()
            undelivered = [
                it for it in states["items"]
                if it is not None
                and not self._item_in_target(it["id"], it["pos"])
            ]
            targets_ml = [x for x in undelivered
                          if x["size"] in ("middle", "large")]
            targets_sm = [x for x in undelivered if x["size"] == "small"]
            # middle/large: pose BOTH robots (co-grab practice);
            # small: pose ONE robot already carrying (transport practice).
            if targets_ml:
                pool, need_partner = targets_ml, True
            elif targets_sm:
                pool, need_partner = targets_sm, False
            else:
                return
            it = pool[int(np.random.randint(len(pool)))]
            item_pos = np.array(it["pos"], dtype=float)
            body = self.get_item_by_id(it["id"]).shape_body
            saved.append((body, tuple(body.position), float(body.angle)))
            # Transport curriculum: teleport the item to a random free pose
            # along the way so ALL transport stages get practice
            # (especially the final approach to the target).
            tele_frac = float(getattr(self, "cograb_teleport_frac", 0.5))
            if np.random.random() < tele_frac:
                r_item = self._item_circumradius(it["id"]) + 0.03
                for _ in range(40):
                    cand = np.random.uniform(-1.0, 1.0, size=2)
                    if any(
                        (min(w[0][0], w[1][0]) - r_item <= cand[0]
                         <= max(w[0][0], w[1][0]) + r_item)
                        and (min(w[0][1], w[1][1]) - r_item <= cand[1]
                             <= max(w[0][1], w[1][1]) + r_item)
                        for w in self.walls if w[0] is not None
                    ):
                        continue
                    d_tg = self.get_distance_to_target_areas(list(cand))
                    if d_tg is None or d_tg < 0.3:
                        continue
                    body.position = tuple(cand)
                    body.velocity = (0.0, 0.0)
                    body.angular_velocity = 0.0
                    # Static bodies keep a stale spatial index when moved
                    # directly — without reindexing, collisions and grab
                    # probes still "see" the item at its old position.
                    self._space.reindex_shapes_for_body(body)
                    self._space.step(1e-6)
                    item_pos = np.array(cand, dtype=float)
                    break
            holder = int(np.random.randint(2))
            placed = []
            rids = [holder, 1 - holder] if need_partner else [holder]
            for k, rid in enumerate(rids):
                ok = False
                r_ring = self._item_circumradius(it["id"])
                for _ in range(60):
                    ang_dir = np.random.uniform(-math.pi, math.pi)
                    d = np.array([math.cos(ang_dir), math.sin(ang_dir)])
                    pos = item_pos + d * (r_ring + np.random.uniform(0.09, 0.14))
                    if abs(pos[0]) > 1.15 or abs(pos[1]) > 1.15:
                        continue
                    if placed and np.linalg.norm(pos - placed[0]) < 0.2:
                        continue
                    r_rob = 0.11
                    if any(
                        (min(w[0][0], w[1][0]) - r_rob <= pos[0]
                         <= max(w[0][0], w[1][0]) + r_rob)
                        and (min(w[0][1], w[1][1]) - r_rob <= pos[1]
                             <= max(w[0][1], w[1][1]) + r_rob)
                        for w in self.walls if w[0] is not None
                    ):
                        continue
                    fwd = -d  # face the item
                    theta = math.atan2(-fwd[0], fwd[1])
                    robot = self._robots[rid]
                    robot.body.position = tuple(pos)
                    robot.body.angle = theta
                    robot.body.velocity = (0.0, 0.0)
                    robot.body.angular_velocity = 0.0
                    self._space.step(1e-6)
                    hit = self.check_for_object_in_front(self._space, robot)
                    if hit is not None and getattr(
                        hit.body, "custom_id", None
                    ) == it["id"]:
                        ok = True
                        break
                if not ok:
                    if k == 1 and self.if_hold[holder]:
                        self.realease_front(self._robots[holder], holder)
                        self.if_hold[holder] = False
                    _rollback()
                    return
                placed.append(np.array(self._robots[rid].body.position))
                if k == 0:
                    self.grab_front(hit, self._robots[rid], rid)
                    self.if_hold[rid] = True
        except Exception:
            try:
                _rollback()
            except Exception:
                pass

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

        # Physics blow-ups (observed on rotation/large-item maps) can drive
        # body positions to NaN; the rest of the episode would be garbage.
        # Truncate immediately so training just moves to the next episode.
        try:
            _pos = [self._robots[0].body.position, self._robots[1].body.position]
            _pos += [s.shape_body.position for s in self._MovingOutEnv__items_shapes]
            if any(
                math.isnan(p[0]) or math.isnan(p[1]) or math.isinf(p[0]) or math.isinf(p[1])
                for p in _pos
            ):
                truncated = True
                info["physics_nan"] = True
        except Exception:
            pass

        info["global_score"] = self.global_score()

        if self.use_state:
            obs = self.get_state()
        if info.get("physics_nan"):
            obs = np.nan_to_num(np.asarray(obs, dtype=np.float64)).tolist()
            rew = [0.0 if (isinstance(r, float) and math.isnan(r)) else r for r in rew]
        return obs, rew, terminated, truncated, info
