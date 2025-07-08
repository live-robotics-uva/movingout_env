import argparse
import json
import math


import numpy as np


def calculate_waiting_time(trajectory):
    conter = 0
    for state in trajectory[0]:
        if (
            (
                state["states"]["robot_1"]["hold"]
                and not state["states"]["robot_2"]["hold"]
                and state["states"]["robot_1"]["holded_item_category"] != "small"
            )
            or (
                state["states"]["robot_2"]["hold"]
                and not state["states"]["robot_1"]["hold"]
                and state["states"]["robot_2"]["holded_item_category"] != "small"
            )
            or (
                state["states"]["robot_1"]["hold"]
                and state["states"]["robot_2"]["hold"]
                and state["states"]["robot_1"]["holded_item_category"] != "small"
                and state["states"]["robot_2"]["holded_item_category"] != "small"
                and state["states"]["robot_1"]["holded_item_id"]
                != state["states"]["robot_2"]["holded_item_id"]
            )
        ):
            conter += 1
    return conter / len(trajectory[0])


def compute_action_consistency(pos1, pos2, action1, action2):
    x1, y1 = pos1
    x2, y2 = pos2
    mag1, dir1 = action1
    mag2, dir2 = action2

    v = np.array([x2 - x1, y2 - y1], dtype=float)
    length = np.linalg.norm(v)

    if length < 1e-9:
        return 1.0

    hat_v = v / length

    f1 = np.array([mag1 * math.cos(dir1), mag1 * math.sin(dir1)], dtype=float)
    f2 = np.array([mag2 * math.cos(dir2), mag2 * math.sin(dir2)], dtype=float)

    p1 = np.dot(f1, hat_v)
    p2 = np.dot(f2, hat_v)

    total = abs(mag1) + abs(mag2)
    if total < 1e-9:
        return 1.0

    if p1 * p2 < 0:
        wasted = abs(p1) + abs(p2)
    else:
        wasted = 0.0

    efficiency = 1.0 - wasted / total
    return efficiency


def calculate_action_consistency(trajectory):
    conter = 0
    counter = 0
    for state in trajectory[0]:
        if (
            state["states"]["robot_1"]["hold"]
            and state["states"]["robot_2"]["hold"]
            and state["states"]["robot_1"]["holded_item_category"] != "small"
            and state["states"]["robot_2"]["holded_item_category"] != "small"
            and state["states"]["robot_1"]["holded_item_id"]
            == state["states"]["robot_2"]["holded_item_id"]
        ):
            pos1 = state["states"]["robot_1"]["pos"]
            pos2 = state["states"]["robot_2"]["pos"]
            action1 = state["action"][0][0:2]
            action2 = state["action"][1][0:2]
            conter += compute_action_consistency(pos1, pos2, action1, action2)
            counter += 1

    return conter / (counter + 1e-5)


def calculate_additional_metrics(trajectory):
    # print("Calculating waiting time...", calculate_waiting_time(trajectory) )
    # print("Calculating action consistency...", calculate_action_consistency(trajectory) )
    return {
        "waiting_time": calculate_waiting_time(trajectory),
        "action_consistency": calculate_action_consistency(trajectory),
    }


def load_json(file_path):
    """Load the JSON data from the specified file."""
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {file_path}.")
        return None


def main():
    # Argument parser to take file path as input
    parser = argparse.ArgumentParser(description="Read and process a JSON file.")
    parser.add_argument("--json_path", type=str, help="Path to the JSON file")

    # Parse arguments
    args = parser.parse_args()

    # Load JSON data from the file
    json_data = load_json(args.json_path)

    calculate_additional_metrics(json_data)


if __name__ == "__main__":
    main()
