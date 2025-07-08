import heapq
import math

import numpy as np


def heuristic(a, b):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def get_cloest_point(xy):
    x = xy[0]
    y = xy[1]

    x = round((x + 1.2) / 0.05)
    y = round((y + 1.2) / 0.05)
    return (x, y)


def astar(array, start, goal, discrete_start_goal=False):
    if discrete_start_goal:
        start = get_cloest_point(start)
        goal = get_cloest_point(goal)

    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data[::-1]

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor) * (
                math.sqrt(2) if i != 0 and j != 0 else 1
            )
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [
                i[1] for i in oheap
            ]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return False


if __name__ == "__main__":
    from discrete_env import discretize_environment

    X_dimensions = np.array([(-1.2, -1.2), (1.2, 1.2)])
    Obstacles = np.array([[[0, 0], [0.4, 0.4]], [[0.2, 0.6], [0.4, 0.8]]])  # Obstacles
    grid_resolution = 0.05
    grid = discretize_environment(X_dimensions, Obstacles, grid_resolution)

    start_point = (0, 0)
    goal_point = (12.0, 12.0)

    import time

    print(time.time())
    for i in range(1000):
        astar(grid, start_point, goal_point)
    print(time.time())
