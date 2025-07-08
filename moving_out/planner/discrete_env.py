import numpy as np

try:
    from . import utils
except:
    import utils


def preprocess_goal_region(goal_regions):
    goal_region_converted = []
    for goal_region in goal_regions:
        goal_region_converted.append(
            [
                [goal_region[0], goal_region[1]],
                [goal_region[0] + goal_region[2], goal_region[1] - goal_region[3]],
            ]
        )
    # goal_region_converted = [[
    #     [goal_region[0][0], goal_region[0][1]],
    #     [goal_region[0][0] + goal_region[0][1], goal_region[0][1] - goal_region[1][1]]
    # ]]
    return goal_region_converted


def discretize_environment(X_dimensions, Obstacles, goal_region, grid_resolution=0.1):
    """
    Discretize the continuous environment into a grid.

    Parameters:
    - X_dimensions: numpy array of shape (2, 2), defines the boundaries of the environment.
      Example: np.array([(-1.2, -1.2), (1.2, 1.2)])
    - Obstacles: numpy array of obstacles, each defined by (x_min, y_min, x_max, y_max).
      Example: np.array([(0.2, 0.2, 0.4, 0.4), (0.2, 0.6, 0.4, 0.8)])
    - grid_resolution: float, the size of each grid cell.

    Returns:
    - grid: 2D numpy array representing the discretized environment.
      0 indicates free space, and 1 indicates an obstacle.
    """
    Obstacles = utils.convert_walls(Obstacles)

    goal_region = preprocess_goal_region(goal_region)
    goal_region = utils.convert_goal_region(goal_region)
    # Extract the environment boundaries
    x_min, y_min = X_dimensions[0]
    x_max, y_max = X_dimensions[1]

    # Calculate the number of grid cells along each dimension
    x_size = int(np.ceil((x_max - x_min) / grid_resolution)) + 1
    y_size = int(np.ceil((y_max - y_min) / grid_resolution)) + 1

    # Initialize the grid with zeros (free space)
    grid = np.zeros((x_size, y_size), dtype=int)

    # Function to check if a grid cell overlaps with any obstacle
    def cell_overlaps_obstacle(cell_min, cell_max, Obstacles):
        if cell_max[0] >= 1.20 and cell_max[1] >= 1.20:
            # print(cell_min, cell_max)
            pass
        for obstacle in Obstacles:
            obs_min = np.array([obstacle[0], obstacle[2]])
            obs_max = np.array([obstacle[1], obstacle[3]])
            # Check for overlap
            if cell_max[0] >= 1.21 or cell_max[1] >= 1.21:
                return True
            if not (
                (cell_max[0] - obs_min[0] <= 0.01)
                or (cell_min[0] - obs_max[0] >= -0.01)
                or (cell_max[1] - obs_min[1] <= 0.01)
                or (cell_min[1] - obs_max[1] >= -0.01)
            ):
                return True
        return False

    def cell_overlaps_goal_region(cell_min, cell_max, goal_regions):
        for goal_region in goal_regions:
            obs_min = np.array([goal_region[0], goal_region[2]])
            obs_max = np.array([goal_region[1], goal_region[3]])
            # Check for overlap
            if cell_max[0] >= 1.21 or cell_max[1] >= 1.21:
                return False
            if not (
                (cell_max[0] - obs_min[0] <= 0.01)
                or (cell_min[0] - obs_max[0] >= -0.01)
                or (cell_max[1] - obs_min[1] <= 0.01)
                or (cell_min[1] - obs_max[1] >= -0.01)
            ):
                return True
        return False

    # Loop through each cell in the grid
    for i in range(x_size):
        for j in range(y_size):
            # Compute the coordinates of the cell
            cell_min = np.array(
                [x_min + i * grid_resolution, y_min + j * grid_resolution]
            )
            cell_max = cell_min + grid_resolution
            # Check if the cell overlaps with any obstacle
            if cell_overlaps_obstacle(cell_min, cell_max, Obstacles):
                grid[i, j] = 1  # Mark as obstacle
            elif cell_overlaps_goal_region(cell_min, cell_max, goal_region):
                grid[i, j] = 2  # Mark as obstacle

    return grid


if __name__ == "__main__":
    # Define the environment dimensions and obstacles
    X_dimensions = np.array([(-1.2, -1.2), (1.2, 1.2)])  # Environment boundaries
    Obstacles = np.array(
        [
            [[0.71, 0.7], [0.77, -1.2]],
            [[0.22, 1.2], [0.28, -0.7]],
            [[-0.27, 0.7], [-0.21, -1.2]],
            [[-0.76, 1.2], [-0.7, -0.7]],
        ]
    )  # Obstacles

    # Discretize the environment
    grid_resolution = 0.05  # Adjust the grid resolution as needed

    goal_region = [[0.2, 0.6, 0.6, 0.6]]
    # goal_region_converted = [[
    #     [goal_region[0][0], goal_region[0][1]],
    #     [goal_region[0][0] + goal_region[0][1], goal_region[0][1] - goal_region[1][1]]
    # ]]
    grid = discretize_environment(X_dimensions, Obstacles, goal_region, grid_resolution)

    # Print the discretized grid
    print("Discretized Grid:")
    print(grid)

    import matplotlib.pyplot as plt

    plt.imshow(
        grid.T,
        origin="lower",
        cmap="Greys",
        extent=(
            X_dimensions[0][0],
            X_dimensions[1][0],
            X_dimensions[0][1],
            X_dimensions[1][1],
        ),
    )
    plt.colorbar(label="Occupancy")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Discretized Environment Grid")
    # plt.show()
    plt.savefig("temp.png")
