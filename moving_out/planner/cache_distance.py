import argparse
import heapq
import json
import math
import multiprocessing as mp
import time

import moving_out.planner.discrete_env as discrete_env
import numpy as np

from moving_out.planner.bfs import bfs
from tqdm import tqdm


def process_path_planning(start_end_pair):
    global grid
    start, end = start_end_pair
    try:
        path = bfs(grid, start, end, goal_is_point=True, discrete_start_goal=True)
    except Exception as e:
        print(f"Error processing path for {start} to {end}: {e}")
        return start, end, [], 0
    if path is False:
        path = []
    # return start, end, len(path), path[:1]
    return start, end, len(path)


def process_path_planning_to_goal_region(start_end_pair):
    global grid
    start, _ = start_end_pair
    try:
        path = bfs(grid, start, None, goal_is_point=False, discrete_start_goal=True)
    except Exception as e:
        print(f"Error processing path for {start} to {_}: {e}")
        return start, _, [], 0
    if path is False:
        path = []
    # return start, _, len(path), path[:1]
    return start, len(path)


def init_worker(g):
    global grid
    grid = g


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_path", type=str, default="")
    args = parser.parse_args()

    with open(args.map_path, "r") as file:
        data = json.load(file)
        walls = data["walls"]
        walls = [x for x in walls if x[0] is not None]
        goal_region = data["target_areas"]
        grid_resolution = 0.05
        X_dimensions = np.array([(-1.2, -1.2), (1.2, 1.2)])
        Obstacles = np.array(walls)
        grid = discrete_env.discretize_environment(
            X_dimensions, Obstacles, goal_region, grid_resolution
        )

    # Generate all combinations of start and end points
    combination = [
        (i, j)
        for i in np.linspace(-1.2, 1.2, num=48)
        for j in np.linspace(-1.2, 1.2, num=48)
    ]
    start_end_combination = [
        (m, n) for i, m in enumerate(combination) for j, n in enumerate(combination)
    ]

    # Multiprocessing setup
    num_workers = mp.cpu_count()  # Use all available CPU cores
    num_workers = 38
    pool = mp.Pool(processes=num_workers, initializer=init_worker, initargs=(grid,))

    # Run the A* pathfinding in parallel
    result_cache = []
    with tqdm(total=len(start_end_combination)) as pbar:
        for result in pool.imap_unordered(process_path_planning, start_end_combination):
            result_cache.append(result)
            pbar.update(1)

    save_file_name = args.map_path.replace("json", "npy")
    result_cache = np.array(result_cache, dtype=object)
    np.save(save_file_name, result_cache, allow_pickle=True)

    pool.close()
    pool.join()

    pool = mp.Pool(processes=num_workers, initializer=init_worker, initargs=(grid,))
    result_cache = []
    with tqdm(total=len(start_end_combination)) as pbar:
        for result in pool.imap_unordered(
            process_path_planning_to_goal_region, start_end_combination
        ):
            result_cache.append(result)
            pbar.update(1)

    save_file_name = args.map_path.replace(".json", "_to_goal_region.npy")
    result_cache = np.array(result_cache, dtype=object)
    np.save(save_file_name, result_cache, allow_pickle=True)

    pool.close()
    pool.join()

    # Save results to file
