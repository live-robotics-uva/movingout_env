from collections import defaultdict


def brute_force_find_common_paths_nested_dict(trajectories, grid_size=(-11, 12)):
    # Initialize the nested dictionary
    path_dict = defaultdict(lambda: defaultdict(list))

    # Generate all possible starting points in the grid
    start_points = [(x, y) for x in range(grid_size[0], grid_size[1]) for y in range(grid_size[0], grid_size[1])]

    # Iterate over all starting points
    for start in start_points:
        # Check all trajectories for paths starting at 'start'
        for traj_id, traj in enumerate(trajectories):
            try:
                start_idx = traj.index(list(start))
            except ValueError:
                continue  # Current trajectory does not contain 'start'

            # Extract the path starting at 'start'
            for end_idx in range(start_idx + 1, len(traj)):
                start_tuple = tuple(traj[start_idx])
                end_tuple = tuple(traj[end_idx])
                path_dict[start_tuple][end_tuple].append(traj_id)

    return path_dict

# Example usage
trajectories = [
    [[-1, 2], [-1, 3], [0, 3], [1, 4]],
    [[-1, 2], [-1, 3], [0, 3], [1, 5]],
    [[-1, 3], [7, 3], [1, 5]]
]

path_dict = brute_force_find_common_paths_nested_dict(trajectories, grid_size=(-11, 12))

# Print the nested dictionary
for start, ends in path_dict.items():
    for end, traj_ids in ends.items():
        print(f"Start: {start}, End: {end}, Trajectories: {traj_ids}")
