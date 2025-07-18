import json
import math
import os

import numpy as np

current_file_path = os.path.dirname(os.path.abspath(__file__))


class StatesEncoder:
    def __init__(
        self,
        if_encode_velocity=True,
        if_encode_object_hold=False,
        encode_wall=False,
        encode_target_area=False,
        max_number_shaps=7,
        encoding="states",
    ):
        self.encoding = encoding
        if self.encoding == "states":
            self.if_encode_velocity = if_encode_velocity
            self.if_encode_object_hold = if_encode_object_hold
            self.encode_wall = encode_wall
            self.encode_target_area = encode_target_area

            self.max_number_shaps = max_number_shaps
        elif self.encoding == "image":
            pass
        self.robot_1_obs_index = None
        self.robot_2_obs_index = None

    def get_robot_obs_index(self):
        # This is not a good implementation; it's only for the convenience of running the code. This implementation assumes that the states of robot_1 and robot_2 are always at the very beginning when encoding the state, and that they will never change.
        if(self.robot_1_obs_index is None):
            print("Please encode state once to initialize the robot obs index")
            return None, None
        else:
            return self.robot_1_obs_index, self.robot_2_obs_index

    def get_objects_feture_by_id(self, id):
        def find_and_read_json_files(id_prefix):
            json_files_data = []

            # Iterate over all files in the folder
            for filename in os.listdir(self.maps_folder_path):
                # Check if the file starts with the specified id and ends with .json
                if filename.startswith(f"{id_prefix}_") and filename.endswith(".json"):
                    file_path = os.path.join(self.maps_folder_path, filename)
                    # Read the JSON file
                    with open(file_path, "r", encoding="utf-8") as file:
                        try:
                            data = json.load(file)
                            json_files_data.append(data)
                        except json.JSONDecodeError:
                            print(
                                f"File {filename} is not a valid JSON file, skipping."
                            )
            valid_objects = [
                x for x in json_files_data[0]["objects"].values() if x is not None
            ]
            return valid_objects

        return find_and_read_json_files(id_prefix=id)

    # def covert_states_to_data_json(self, encoded_states, json_template):

    #     if(not self.init_components):
    #         self.init_components_length(json_template)
    #     robot_state_len = len(self.robot_1
    #     self.

    #     return json

    # def get_image_obs_by_current_obs_states(self, states, map_name):
    #     from moving_out.benchmarks.moving_out import MovingOutEnv
    #     env = MovingOutEnv(use_state=True, map_name = json_data[0][0]["id"])

    def get_state_by_current_obs_states(self, states):

        robot_1 = states["robot_1"]
        robot_2 = states["robot_2"]

        robot_1_states = self.encode_robot(robot_1)
        robot_2_states = self.encode_robot(robot_2)

        self.robot_1_obs_index = [0, len(robot_1_states) - 1]
        self.robot_2_obs_index = [len(robot_1_states), len(robot_1_states) + len(robot_2_states) - 1]

        objects_states_encoding = []
        objects = states["items"]

        for i, sp in enumerate(objects):
            objects_states_encoding += self.encode_objects(sp, None)

        if len(objects) < self.max_number_shaps:
            objects_states_encoding += (
                [0]
                * (self.max_number_shaps - len(objects))
                * int(len(objects_states_encoding) / len(objects))
            )

        walls = states["walls"]
        if self.encode_wall:
            walls_encoding = self.encoder_walls(walls)
        else:
            walls_encoding = []
        # walls_encoding = self.encoder_walls(walls)
        if self.encode_target_area:
            target_area = states["target_area"]
        else:
            target_area = []
        return [
            robot_1_states
            + robot_2_states
            + objects_states_encoding
            + walls_encoding
            + target_area,
            robot_2_states
            + robot_1_states
            + objects_states_encoding
            + walls_encoding
            + target_area,
        ]

    def get_state_by_json(self, json_data):
        # id = json_data["id"]
        # original_objects_in_map = self.get_objects_feture_by_id(id)
        # steps = json_data["step"]
        states = json_data["states"]

        return self.get_state_by_current_obs_states(states)

    def encode_robot(self, robot):
        pos = robot["pos"]
        angle = robot["angle"]
        angle = [math.cos(angle), math.sin(angle)]
        hold = robot["hold"]

        pos_encoding = pos
        angle_encoding = angle
        hold_encoding = [1] if hold else [-1]

        if self.if_encode_velocity:
            velocity = robot["velocity"]
            angular_velocity = robot["angular_velocity"]
            velocity_encoding = velocity
            angular_velocity_encoding = [angular_velocity]
            return (
                pos_encoding
                + angle_encoding
                + hold_encoding
                + velocity_encoding
                + angular_velocity_encoding
            )
        else:
            return pos_encoding + angle_encoding + hold_encoding

    def shape_onehot_encoding(self, shape):
        mapping = {
            "triangle": [1, 0, 0, 0, 0, 0, 0, 0],
            "square": [0, 1, 0, 0, 0, 0, 0, 0],
            "pentagon": [0, 0, 1, 0, 0, 0, 0, 0],
            "hexagon": [0, 0, 0, 1, 0, 0, 0, 0],
            "octagon": [0, 0, 0, 0, 1, 0, 0, 0],
            "circle": [0, 0, 0, 0, 0, 1, 0, 0],
            "star": [0, 0, 0, 0, 0, 0, 1, 0],
            "rectangle": [0, 0, 0, 0, 0, 0, 0, 1],
        }

        return mapping[shape]

    def size_onehot_encoding(self, size):
        mapping = {"small": [1, 0, 0], "middle": [0, 1, 0], "large": [0, 0, 1]}
        return mapping[size]

    def encoder_walls(self, walls):
        no_wall = [0, 0, 0, 0]
        encoding = []
        max_walls = 5
        if len(walls) < max_walls:
            walls += [[None]] * (max_walls - len(walls))
        for wall in walls:
            if len(wall) == 1:
                wall = wall[0]
            if wall is not None:
                if len(wall) == 2:
                    wall = wall[0] + wall[1]

            if wall is None:
                encoding += no_wall
            else:
                encoding += wall
        return encoding

    def if_action_should_be_filted(self, current_action, next_action):
        if abs(current_action[0]) <= 0.01 and abs(next_action[0]) <= 0.01:
            if (
                abs(current_action[1] - next_action[1]) <= 0.01
                and current_action[2] == next_action[2]
            ):
                return True
        return False

    def encode_objects(self, movable_object, original_shape_feature):
        pos = movable_object["pos"]
        # angle = (movable_object["angle"] + np.pi) % (2 * np.pi) - np.pi
        angle = movable_object["angle"]
        angle = [math.cos(angle), math.sin(angle)]
        if self.if_encode_object_hold:
            hold = movable_object["hold"]
            hold_encoding = [1 if x else -1 for x in hold]
        else:
            hold_encoding = []
        try:
            shape = movable_object["shape"]
            size = movable_object["size"]
        except:
            shape = original_shape_feature["shape"]
            size = original_shape_feature["size"]

        pos_encoding = pos
        angle_encoding = angle

        shape_encoding = self.shape_onehot_encoding(shape)

        size_encoding = self.size_onehot_encoding(size)


        return (
            pos_encoding
            + angle_encoding
            + hold_encoding
            + shape_encoding
            + size_encoding
        )


if __name__ == "__main__":
    with open(
        r"C:\onedrive\OneDrive - University of Virginia\MOVINGOUT_DATASET\non_expert\1000_20240926174118.json"
    ) as user_file:
        file_contents = user_file.read()
    parsed_json = json.loads(file_contents)[0]

    states_encoder = StatesEncoder()
    states_encoding = states_encoder.get_state_by_json(parsed_json[0])
    print(len(states_encoding[0]))
