import argparse
import os
import time

import cv2
import pygame
from moving_out.benchmarks.moving_out import MovingOutEnv
from moving_out.env_parameters import AVAILABLE_MAPS
from moving_out.utils.data_collection import DataCollector
from moving_out.utils.joystick_controller import JoystickController
from moving_out.utils.keyboard_controller import KeyboardController
from moving_out.utils.utils import append_step_to_file, init_trajectory_file


class BaseCollector(DataCollector):
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir
        self._started = True
        self._finish_early = False
        self._last_image = None
        self.start_moving = [True, True]

    def imshow(self, image):
        self._last_image = image
        cv2.imshow("MovingOut", image)
        cv2.waitKey(1)

    def _init_joysticks(self):
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            print("No joystick detected")
        self.joysticks = []
        for i in range(joystick_count):
            joystick = pygame.joystick.Joystick(i)
            joystick.init()
            self.joysticks.append(joystick)
            print(f"Detected {i}: {joystick.get_name()}")

    def run_data_collection_loop(
        self,
        dt,
        env,
        map_id,
        step_fn,
        ids=None,
        red_name="",
        blue_name="",
        reverse_action=False,
        test_mode=False,
        first_time_reverse=False,
    ):
        pygame.init()
        self.reverse_action = reverse_action
        if hasattr(self, "joystick_event_process"):
            self._init_joysticks()

        env.reset(map_name=map_id, add_noise_to_item=False)

        last_time = time.time()
        rew = 0
        done = False
        steps = 0

        json_file_path = init_trajectory_file(map_id, self.save_dir)
        states = env.get_all_states()
        steps_data = []

        obs = env.render("rgb_array")
        obs = self.display_image_with_overlay(obs, 0, 0)
        self.imshow(obs)

        if hasattr(self, "joystick_event_process") and hasattr(self, "start_moving"):
            self.joystick_event_process()
            while sum(self.start_moving) != 2:
                print("Please start moving both joysticks, ", self.start_moving)
                time.sleep(0.1)
                self.joystick_event_process()

        time_0 = time.time()

        while not self._finish_early:
            steps += 1
            time_1 = time.time()
            if steps % 10 == 0:
                print(f"Time: {time_1 - time_0:.3f}")

            if hasattr(self, "joystick_event_process"):
                self.joystick_event_process()

            action = self.get_action()

            step_data = {
                "red_name": red_name,
                "blue_name": blue_name,
                "time": time.strftime("%Y-%m-%d_%H:%M:%S"),
                "id": map_id,
                "step": steps,
                "states": states,
                "rew": rew,
                "done": done,
                "action": action,
                "test_mode": test_mode,
            }
            steps_data.append(step_data)

            if self._started:
                obs, rew, done, info = step_fn(action)
                obs = self.display_image_with_overlay(
                    obs, info["eval_score"], time_1 - time_0
                )
                states = env.get_all_states()
                if done or info["eval_score"] >= 1.0:
                    self.imshow(obs)
                    append_step_to_file(json_file_path, steps_data)
                    return

                if obs is None:
                    return
                self.imshow(obs)
            else:
                self.imshow(self._last_image)

            delta = time.time() - last_time
            time.sleep(max(0, dt - delta))
            last_time = time.time()


class JoystickDataCollector(BaseCollector, JoystickController):
    def __init__(self, save_dir):
        BaseCollector.__init__(self, save_dir)
        JoystickController.__init__(self)

    def get_action(self):
        return self.get_actions()


class KeyboardDataCollector(BaseCollector, KeyboardController):
    def __init__(self, save_dir):
        BaseCollector.__init__(self, save_dir)
        KeyboardController.__init__(self)

    def get_action(self):
        action1, action2 = self.get_actions()
        return [action1, action2]


def _parse_map_id(raw_value):
    if raw_value is None:
        return None
    if isinstance(raw_value, str) and raw_value in AVAILABLE_MAPS:
        return raw_value

    return None


def _prompt_map_id():
    string_maps = sorted(key for key in AVAILABLE_MAPS if isinstance(key, str))
    print("Available maps:")
    print(", ".join(string_maps))
    user_input = input("Select map name: ").strip()
    map_id = _parse_map_id(user_input)
    if map_id is None:
        raise ValueError(f"Invalid map id: {user_input}")
    return map_id


def _prompt_input_mode():
    user_input = input("Select input (joystick/keyboard): ").strip().lower()
    if user_input not in {"joystick", "keyboard"}:
        raise ValueError(f"Invalid input mode: {user_input}")
    return user_input


def _prompt_num():
    user_input = input("Number of collections: ").strip()
    try:
        num = int(user_input)
    except ValueError:
        raise ValueError(f"Invalid number: {user_input}")
    if num < 1:
        raise ValueError("Number of collections must be >= 1")
    return num


def run_collection(map_id, num_collect, input_mode, output_dir):
    env = MovingOutEnv(use_state=False, map_name=map_id)
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    if input_mode == "keyboard":
        collector = KeyboardDataCollector(output_dir)
    else:
        collector = JoystickDataCollector(output_dir)

    def step(action):
        obs, rew, _, done, info = env.step(action)
        rgb_obs = env.render("rgb_array")
        return rgb_obs, rew, done, info

    for _ in range(num_collect):
        collector.run_data_collection_loop(
            1 / 10.0,
            env,
            map_id,
            step,
            ids=map_id,
            red_name="",
            blue_name="",
            reverse_action=False,
            test_mode=False,
            first_time_reverse=False,
        )


def main():
    parser = argparse.ArgumentParser(description="Collect MovingOut joystick data.")
    parser.add_argument("--map", type=str, help="Map id or name.")
    parser.add_argument("--num", type=int, help="Number of collections.")
    parser.add_argument(
        "--input",
        type=str,
        choices=["joystick", "keyboard"],
        help="Input device to use.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="saved_datasets",
        help="Output directory for trajectory files.",
    )
    args = parser.parse_args()

    map_id = _parse_map_id(args.map)
    if map_id is None:
        map_id = _prompt_map_id()

    input_mode = args.input or _prompt_input_mode()
    num_collect = args.num if args.num is not None else _prompt_num()

    if num_collect < 1:
        raise ValueError("--num must be >= 1")

    run_collection(map_id, num_collect, input_mode, args.output)


if __name__ == "__main__":
    main()