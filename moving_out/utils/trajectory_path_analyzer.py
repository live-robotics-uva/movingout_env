import copy
import json
from collections import defaultdict

import numpy as np
from moving_out.utils.discretizer import Discretizer
from tqdm import tqdm


class TrajectoryPathAnalyzer:
    def __init__(self, grid_size=(-11, 12)):
        self.grid_size = grid_size
        self.trajectories = []
        self.path_dict = defaultdict(lambda: defaultdict(list))

    def load_trajectories(self, file_path):
        """Load trajectory data from a JSON file."""
        with open(file_path, "r") as f:
            self.trajectories = json.load(f)

    def compute_from_data_file(self, file_path):
        loaded_data = np.load(file_path, allow_pickle=True)
        discreted_data = copy.deepcopy(loaded_data)
        trajectories = []
        discretizer = Discretizer(-1.2, 1.2, -23, 24)
        for i in range(len(loaded_data)):
            trajectory = []
            for j in range(len(loaded_data[i])):
                discreted_data[i][j][0][0:2] = discretizer.discretize(
                    loaded_data[i][j][0][0:2]
                )
                trajectory.append(discreted_data[i][j][0][0:2])
            trajectories.append(trajectory)
        self.trajectories = trajectories
        self.compute_path_dict()

    def compute_path_dict(self):
        """Compute the path_dict based on loaded trajectories."""
        self.path_dict = defaultdict(lambda: defaultdict(list))
        for traj_id, traj in enumerate(tqdm(self.trajectories)):
            for start_idx in range(len(traj) - 1):
                start = tuple(traj[start_idx])
                for end_idx in range(start_idx + 1, len(traj)):
                    end = tuple(traj[end_idx])
                    manhattan_dist = abs(start[0] - traj[end_idx][0]) + abs(
                        start[1] - traj[end_idx][1]
                    )
                    if manhattan_dist > 20 or end_idx - start_idx != 8:
                        continue
                    if traj_id not in self.path_dict[start][end]:
                        self.path_dict[start][end].append(
                            [traj_id, end_idx - start_idx, start_idx, end_idx]
                        )

    def save_cache(self, cache_path):
        """Save the computed path_dict to a file."""
        # Convert tuple keys to string for JSON compatibility
        path_dict_serializable = {
            str(k): {str(ek): v for ek, v in e.items()}
            for k, e in self.path_dict.items()
        }
        with open(cache_path, "w") as f:
            json.dump(path_dict_serializable, f)

    def load_cache(self, cache_path):
        """Load the path_dict from a file."""
        with open(cache_path, "r") as f:
            path_dict_serializable = json.load(f)
        # Convert string keys back to tuple
        self.path_dict = {
            tuple(map(int, k.strip("()").split(","))): {
                tuple(map(int, ek.strip("()").split(","))): v for ek, v in e.items()
            }
            for k, e in path_dict_serializable.items()
        }

    def query_by_start_and_end(self, start, end):
        """Query the path_dict for a given start and end point."""
        start_tuple = tuple(start)
        end_tuple = tuple(end)
        if start_tuple in self.path_dict and end_tuple in self.path_dict[start_tuple]:
            return {
                "start": start_tuple,
                "end": end_tuple,
                "trajectories": self.path_dict[start_tuple][end_tuple],
            }
        else:
            return {"start": start_tuple, "end": end_tuple, "trajectories": []}


# Example Usage
if __name__ == "__main__":
    analyzer = TrajectoryPathAnalyzer(grid_size=(-11, 12))

    # Load trajectories from a file
    analyzer.load_trajectories("trajectories.json")  # Replace with actual file path

    # Compute the path dictionary
    analyzer.compute_path_dict()

    # Save the computed dictionary to cache
    analyzer.save_cache("path_dict_cache.json")

    # Load the dictionary from cache
    analyzer.load_cache("path_dict_cache.json")

    # Query the dictionary for a specific start and end point
    start_point = [-1, 3]
    end_point = [1, 5]
    result = analyzer.query_by_start_and_end(start_point, end_point)

    # Print the result
    print(
        f"Start: {result['start']}, End: {result['end']}, Trajectories: {result['trajectories']}"
    )
