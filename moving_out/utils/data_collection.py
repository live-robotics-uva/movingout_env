import pygame
import time
import os
from datetime import datetime
from .utils import append_step_to_file, init_trajectory_file, resultant_vector
import numpy as np
import cv2
import json
class DataCollector:
    def __init__(self):
        pass
    
    def append_step_to_file(self, file_path, step_data):
        with open(file_path, "r+") as f:
            data = json.load(f)
            data.append(step_data)
            f.seek(0)
            json.dump(data, f, indent=4)
    def display_image_with_overlay(self, image, progress, current_time):
        """
        Display an image with time (seconds and milliseconds) and a progress bar overlay.

        Args:
            image (numpy.ndarray): The original image (512, 512, 3).
            progress (float): A value between 0 and 1 representing the progress.
            current_time (float): The current time in seconds (can include fractional seconds).
        """
        if not (0 <= progress <= 1):
            raise ValueError("Progress must be between 0 and 1.")

        # Extract seconds and milliseconds from the current time
        seconds = int(current_time)
        milliseconds = int((current_time * 1000) % 1000)

        # Create an overlay for additional information below the image
        overlay_height = 100
        overlay_width = image.shape[1]  # Same as image width
        overlay = np.zeros((overlay_height, overlay_width, 3), dtype=np.uint8)

        # Add time text overlay
        time_text = f"Time: {seconds}s {milliseconds}ms"
        cv2.putText(
            overlay,
            time_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Add progress bar overlay
        progress_bar_width = overlay_width - 20  # Adjust to fit within the overlay
        progress_bar_height = 20
        progress_bar_x = 10
        progress_bar_y = 50

        # Draw the background of the progress bar
        cv2.rectangle(
            overlay,
            (progress_bar_x, progress_bar_y),
            (progress_bar_x + progress_bar_width, progress_bar_y + progress_bar_height),
            (50, 50, 50),
            -1,
        )

        # Draw the filled portion of the progress bar
        filled_width = int(progress * progress_bar_width)
        cv2.rectangle(
            overlay,
            (progress_bar_x, progress_bar_y),
            (progress_bar_x + filled_width, progress_bar_y + progress_bar_height),
            (0, 255, 0),
            -1,
        )

        # Add progress text overlay
        progress_text = f"Progress: {progress * 100:.1f}%"
        cv2.putText(
            overlay,
            progress_text,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Add black padding to the right of the image to make it square
        padding_width = image.shape[0] - image.shape[1]
        if padding_width > 0:
            padding = np.zeros((image.shape[0], padding_width, 3), dtype=np.uint8)
            image_with_padding = np.hstack((image, padding))
        else:
            image_with_padding = image

        # Combine the image with padding and the overlay
        combined_image = np.vstack((image_with_padding, overlay))

        # Show the image with overlay
        return combined_image



    def run_data_collection_loop(
        self,
        dt,
        env,
        id,
        step_fn,
        ids=[],
        red_name="",
        blue_name="",
        reverse_action=False,
        test_mode=False,
        first_time_reverse=False,
    ):
        """Run an environment interaction loop.

        The step_fn will be continually called with actions, and it should
        return observations. When step_fn returns None, the loop is done.
        """

        pygame.init()
        pygame.joystick.init()
        self.reverse_action = reverse_action

        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            print("No joystick detected")
            # sys.exit()
        joysticks = []
        for i in range(joystick_count):
            joystick = pygame.joystick.Joystick(i)
            joystick.init()
            joysticks.append(joystick)
            print(f"Detected {i}: {joystick.get_name()}")

        env.reset(map_name=id, add_noise_to_item=True)

        last_time = time.time()
        self._started = True

        rew = 0
        done = False
        steps = 0

        save_dir = r".\saved_datasets_1031"
        try:
            os.makedirs(save_dir)
        except:
            pass
        json_file_path = init_trajectory_file(id, save_dir)

        states = env.get_all_states()
        # print(states)
        steps_data = []
        collection_count = 0
        obs = env.render("rgb_array")
        obs = self.display_image_with_overlay(obs, 0, 0)
        self.imshow(obs)
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
            self.joystick_event_process()
            action = self.get_action()

            step_data = {
                "red_name": red_name,
                "blue_name": blue_name,
                "time": datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                "id": id,
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

                    time_0 = time.time()
                    collection_count += 1
                    print(f"COLLECTED: {collection_count}")
                    append_step_to_file(json_file_path, steps_data)
                    return

                if obs is None:
                    return
                self.imshow(obs)

            else:
                pass
                # Needed to run the event loop.
                self.imshow(self._last_image)
            delta = time.time() - last_time
            time.sleep(max(0, dt - delta))
            last_time = time.time()
