import datetime
import json
import math
import os

import numpy as np
from moving_out.env_parameters import AVAILABLE_MAPS, DEFAULT_MAP_PATH
from moving_out.maps.add_noise_to_maps import add_noise_to_map


class states_buffer:
    def __init__(self, length=1) -> None:
        self.length = length
        self.states = []

    def get_states(self):
        return self.states

    def push_states(self, state):
        if self.length == 0:
            return None
        while len(self.states) != self.length:
            self.states.append(state)
        else:
            del self.states[0]
            self.states.append(state)


def append_step_to_file(file_path, step_data):
    with open(file_path, "r+") as f:
        data = json.load(f)
        data.append(step_data)
        f.seek(0)
        json.dump(data, f, indent=4)


def init_trajectory_file(id, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    file_name = f"{id}_{current_time}.json"
    file_path = os.path.join(save_dir, file_name)

    with open(file_path, "w") as f:
        json.dump([], f)

    return file_path


def calculate_average_results(evaluation_results):
    rewards = []
    steps = []
    shapes = [[], [], []]
    global_dense_rewards = []
    average_predictor_score = []
    for i in evaluation_results.values():
        rewards.append(float(i["global_reward"]))
        steps.append(float(i["steps"]))
        global_dense_rewards.append(float(i["global_dense_reward"]))
        average_predictor_score.append(i["predictor"])
        ovl_small = i["overlapped_items"]["small"]
        ovl_middle = i["overlapped_items"]["middle"]
        ovl_large = i["overlapped_items"]["large"]

        if ovl_small[1] == 0:
            pass
        else:
            shapes[0].append(ovl_small[0] / ovl_small[1])

        if ovl_middle[1] == 0:
            pass
        else:
            shapes[1].append(ovl_middle[0] / ovl_middle[1])

        if ovl_large[1] == 0:
            pass
        else:
            shapes[2].append(ovl_large[0] / ovl_large[1])

    return {
        "rewards": [np.mean(rewards), np.std(rewards)],
        "global_dense_rewards": [
            np.mean(global_dense_rewards),
            np.std(global_dense_rewards),
        ],
        "steps": [np.mean(steps), np.std(steps)],
        "small": None
        if len(shapes[0]) == 0
        else [np.mean(shapes[0]), np.std(shapes[0])],
        "middle": None
        if len(shapes[1]) == 0
        else [np.mean(shapes[1]), np.std(shapes[1])],
        "large": None
        if len(shapes[2]) == 0
        else [np.mean(shapes[2]), np.std(shapes[2])],
        "predictor": [
            list(np.array(average_predictor_score).mean(axis=0)),
            list(np.array(average_predictor_score).std(axis=0)),
        ],
    }


def resultant_vector(x, y):
    # Extract start and end points from input
    x_start, x_end = x
    y_start, y_end = y

    # Calculate the components of each vector
    x1 = x_end[0] - x_start[0]
    y1 = x_end[1] - x_start[1]
    x2 = y_end[0] - y_start[0]
    y2 = y_end[1] - y_start[1]

    # Sum the components to get the resultant vector's components
    resultant_x = x1 + x2
    resultant_y = y1 + y2

    # Calculate the magnitude of the resultant vector
    resultant_magnitude = math.sqrt(resultant_x**2 + resultant_y**2)
    # Calculate the angle of the resultant vector with the x-axis
    resultant_angle = math.atan2(resultant_y, resultant_x)

    return resultant_magnitude, resultant_angle


def resultant_vector_by_direction(x, y):
    magnitude, angle = resultant_vector([[0, 0], [x, 0]], [[0, 0], [0, y]])
    return [magnitude, angle]


def reset_env_to_id(env, map_name, add_noise_to_item=None):
    # folder_path = r"C:\Users\Administrator\OneDrive - University of Virginia\github\moving_out_AI\xmagical\maps\all_maps_items"
    env.map_name = map_name
    map_path = os.path.join(DEFAULT_MAP_PATH, AVAILABLE_MAPS[map_name])
    json_data = read_json_files(map_path)
    if add_noise_to_item:
        json_data = add_noise_to_map(json_data, map_name)
    config = json_data["0"]
    env.on_reset(
        robot_1_pos=config["robot_1_pos"],
        robot_1_angle=config["robot_1_angle"],
        robot_2_pos=config["robot_2_pos"],
        robot_2_angle=config["robot_2_angle"],
        walls=config["walls"],
        objects=config["objects"],
        target_areas=config["target_areas"],
        target_color=config["target_color"],
    )


def clone_env_by_actions(env, actions):
    from moving_out.benchmarks.moving_out import MovingOutEnv

    cloned_env = MovingOutEnv(
        use_state=env.use_state,
        reward_setting=env.reward_setting,
        rand_layout_full=env.rand_layout_full,
        rand_shapes=env.rand_shapes,
        rand_colors=env.rand_colors,
        arena_size=env.arena_size,
        map_name=env.map_name,
        dense_rewards_setting=env.dense_rewards_setting,
    )

    for a in actions:
        cloned_env.step(a)

    return cloned_env


def read_json_files(map_path):
    with open(map_path, "r") as file:
        data = json.load(file)

    return data
