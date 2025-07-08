import json
import random

import numpy as np


def add_noise_and_clip(value, noise_scale, bound):
    # Calculate the range of noise
    noise_range = noise_scale

    # Add random noise within the range
    noisy_value = value + random.uniform(-noise_range, noise_range)
    if bound != None:
        # Clip the value to the bound
        noisy_value = max(-bound, min(noisy_value, bound))
    return noisy_value


def add_noise_to_robot_pose(map_config):
    robot_1_position = map_config["0"]["robot_1_pos"]

    robot_1_position[0] = add_noise_and_clip(robot_1_position[0], 0.1, 1.1)
    robot_1_position[1] = add_noise_and_clip(robot_1_position[1], 0.1, 1.1)

    robot_1_angle = map_config["0"]["robot_1_angle"]

    robot_1_angle = random.uniform(-np.pi, np.pi)

    robot_2_position = map_config["0"]["robot_2_pos"]

    robot_2_position[0] = add_noise_and_clip(robot_2_position[0], 0.1, 1.1)
    robot_2_position[1] = add_noise_and_clip(robot_2_position[1], 0.1, 1.1)

    robot_2_angle = random.uniform(-np.pi, np.pi)

    robot_2_angle = add_noise_and_clip(robot_2_angle, 0.1, 2 * np.pi)

    map_config["0"]["robot_1_pos"] = robot_1_position
    map_config["0"]["robot_1_angle"] = robot_1_angle
    map_config["0"]["robot_2_pos"] = robot_2_position
    map_config["0"]["robot_2_angle"] = robot_2_angle

    return map_config


def add_noise_to_item_pose(map_config):
    items = map_config["0"]["objects"]
    for k, v in items.items():
        if v is None:
            continue
        items[k]["pos"][0] = add_noise_and_clip(v["pos"][0], 0.1, 1.1)
        items[k]["pos"][1] = add_noise_and_clip(v["pos"][1], 0.1, 1.1)
        items[k]["angle"] = random.uniform(-np.pi, np.pi)
    map_config["0"]["objects"] = items

    return map_config


def add_noise_to_item_shape(map_config, map_name):
    items = map_config["0"]["objects"]
    can_not_repeat = []
    if int(map_name) >= 2000 and int(map_name) <= 2003:
        for k, v in items.items():
            if v is None:
                continue
            can_not_repeat.append(v["shape"] + v["size"])
    else:
        for k, v in items.items():
            if v is None:
                continue
            can_not_repeat.append(v["shape"] + v["size"])
    possible_shapes = [
        "triangle",
        "square",
        "pentagon",
        "hexagon",
        "octagon",
        "star",
        "rectangle",
    ]
    possible_sizes = ["middle", "large"]

    for k, v in items.items():
        if v is None:
            continue

        if int(map_name) >= 2001 and int(map_name) <= 2003:
            random_shape_n_size = v["shape"] + v["size"]
            while random_shape_n_size in can_not_repeat:
                random_shape = random.choice(possible_shapes)
                if v["size"] == "middle" or v["size"] == "large":
                    random_size = random.choice(possible_sizes)
                    if random_size == "middle":
                        items[k]["shape_scale"] = 0.11
                    elif random_size == "large":
                        items[k]["shape_scale"] = 0.20
                else:
                    random_size = v["size"]
                # random_size = random.choice(possible_sizes)
                random_shape_n_size = random_shape + random_size
                # sum of large items should be less than 2
                # for k, v in items.items():
                #     if(v is None):
                #         continue
                #     if(v["size"] == "large"):
                #         large_number += 1
            items[k]["shape"] = random_shape
            items[k]["size"] = random_size
        else:
            random_shape_n_size = v["shape"] + v["size"]
            while random_shape_n_size in can_not_repeat:
                random_shape = random.choice(possible_shapes)
                random_shape_n_size = random_shape + v["size"]
            items[k]["shape"] = random_shape
            if map_name == 2005 and items[k]["size"] == "large":
                items[k]["shape_scale"] = 0.13

    map_config["0"]["objects"] = items

    return map_config


def add_noise_to_item_size(map_config):
    items = map_config["0"]["objects"]
    for k, v in items.items():
        if v is None:
            continue
        items[k]["shape_scale"] = add_noise_and_clip(
            v["shape_scale"], v["shape_scale"] * 0.1, None
        )
    map_config["0"]["objects"] = items

    return map_config


def add_noise_to_item_mass(map_config):
    items = map_config["0"]["objects"]
    for k, v in items.items():
        if v is None:
            continue
        items[k]["mass"] = add_noise_and_clip(v["mass"], v["mass"] * 0.1, None)
    map_config["0"]["objects"] = items
    return map_config


def add_noise_to_map(map_config, map_name):
    map_config = add_noise_to_robot_pose(map_config)
    map_config = add_noise_to_item_pose(map_config)
    map_config = add_noise_to_item_shape(map_config, map_name)
    map_config = add_noise_to_item_size(map_config)
    map_config = add_noise_to_item_mass(map_config)
    return map_config

    # def main():
    #     with open(r".\maps_v2\1000_ROOM.json", "r") as f:
    #         map_config = json.load(f)
    #     map_config = add_noise_to_robot_pose(map_config)
    #     map_config = add_noise_to_item_pose(map_config)
    #     map_config = add_noise_to_item_shape(map_config)
    #     map_config = add_noise_to_item_size(map_config)
    #     map_config = add_noise_to_item_mass(map_config)

    #     print(map_config["0"])

    # if __name__ == "__main__":
    main()
