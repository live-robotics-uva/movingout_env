import argparse
import json
import time

import cv2
from moving_out.benchmarks.moving_out import MovingOutEnv


def load_json(file_path):
    """Load the JSON data from the specified file."""
    with open(file_path, "r") as file:
        return json.load(file)


def replay_trajectory(json_path, fps):
    json_data = load_json(json_path)
    if not isinstance(json_data, list) or not json_data:
        raise ValueError("Invalid trajectory format: expected a non-empty list.")

    if isinstance(json_data[0], list):
        if not json_data[0]:
            raise ValueError("Invalid trajectory format: empty episode list.")
        steps = json_data[0]
    else:
        steps = json_data

    first_step = steps[0]
    if "id" not in first_step or "action" not in first_step:
        raise ValueError("Invalid trajectory format: missing 'id' or 'action'.")

    map_id = first_step["id"]
    env = MovingOutEnv(use_state=False, map_name=map_id)

    delay_ms = max(1, int(1000 / fps))
    for step in steps:
        action = step["action"]
        obs, _, _, done, info = env.step(action)
        rgb_obs = env.render("rgb_array")
        bgr_obs = cv2.cvtColor(rgb_obs, cv2.COLOR_RGB2BGR)
        cv2.imshow("Replay", bgr_obs)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord("q"):
            break
        if done or info.get("eval_score", 0) >= 1.0:
            break
        if obs is None:
            break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Replay a collected trajectory.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to JSON.")
    parser.add_argument("--fps", type=int, default=10, help="Replay FPS.")
    args = parser.parse_args()

    if args.fps < 1:
        raise ValueError("--fps must be >= 1")

    replay_trajectory(args.json_path, args.fps)


if __name__ == "__main__":
    main()