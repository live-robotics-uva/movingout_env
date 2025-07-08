import math
from collections import deque

import numpy as np


def get_cloest_point(xy):
    x = xy[0]
    y = xy[1]
    x = round((x + 1.2) / 0.05)
    y = round((y + 1.2) / 0.05)

    return (x, y)


def bfs(array, start, goal, goal_is_point, discrete_start_goal=False):
    # Discretize start/goal if requested
    if discrete_start_goal:
        start = get_cloest_point(start)
        if goal_is_point:
            goal_point = get_cloest_point(goal)
    else:
        if goal_is_point:
            goal_point = goal
    # Define neighbor moves (4-connected and diagonal moves)
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    # Initialize the queue with the starting point
    queue = deque([start])
    # Dictionary to reconstruct the path later
    came_from = {}
    # Set to keep track of visited nodes
    visited = {start}

    while queue:
        current = queue.popleft()
        if goal_is_point:
            is_goal = current[0] == goal_point[0] and current[1] == goal_point[1]
        else:
            is_goal = array[current[0]][current[1]] == 2
        # print(is_goal)
        if is_goal:
            # Reconstruct path by following came_from pointers
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            # Check grid bounds
            if 0 <= neighbor[0] < array.shape[0] and 0 <= neighbor[1] < array.shape[1]:
                # Check for obstacles (grid value 1 means obstacle)
                if array[neighbor[0]][neighbor[1]] == 1:
                    continue
                # If the neighbor has not been visited, add it to the queue
                if neighbor not in visited:
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    queue.append(neighbor)
    return False


if __name__ == "__main__":
    from discrete_env import discretize_environment

    # Define the environment dimensions and obstacles as in the A* example
    X_dimensions = np.array([(-1.2, -1.2), (1.2, 1.2)])
    Obstacles = np.array([[[0, 0], [0.2, 0.2]]])
    goal_region = [[-0.8, 0.6, 0.6, 0.6]]

    goal_point = [7, 30]

    grid_resolution = 0.05
    grid = discretize_environment(X_dimensions, Obstacles, goal_region, grid_resolution)

    start_point = (10, 10)

    print(grid.shape)
    import time

    print(time.time())
    for i in range(1):
        print(bfs(grid, start_point, goal_region, goal_is_point=False))
        print(bfs(grid, start_point, goal_point, goal_is_point=True))
    print(time.time())
