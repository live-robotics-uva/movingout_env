# MovingOut Environment

MovingOut Environment, a physically grounded human-AI collaboration

---

## Installation

Set up the environment using conda and pip:

```bash
conda create -n movingout_test_git python=3.10
conda activate movingout_test_git
git clone git@github.com:live-robotics-uva/moving_out_env.git
cd moving_out_env
pip install -e .
```

---

## Running the Environment

### Option 1: Keyboard Control (for basic testing)

```bash
python ./run_env_keyboard.py --map_name HandOff
```

### Option 2: Joystick Control (recommended)

```bash
python ./run_env_joystick.py --map_name HandOff
```

---

## Control Guide

### Keyboard Controls (Testing Only)

**Agent 1:**

* `WASD`: Move up, left, down, right
* `H` (hold): Move backwards
* `Spacebar`: Pick up or drop item

**Agent 2:**

* `Arrow Keys`: Move up, left, down, right
* `Ctrl` (hold): Move backwards
* `Enter`: Pick up or drop item

### Joystick Controls (Recommended)

* **Left Stick**: Move the agent
* **R Button**: Pick up or drop item
* **ZL (hold)**: Move backwards

> **Note:** Keyboard controls are limited in direction and speed. For a smoother and more complete gameplay experience, joystick control is strongly recommended.

---

## Custom Maps

To add a custom map:

1. Refer to the existing map definitions in `moving_out/maps/maps_v1`.
2. Create a new `.json` file describing your custom map.
3. Add the new map name to `AVAILABLE_MAPS` in `moving_out/env_parameters.py`.

---

For issues or contributions, feel free to open an issue or pull request on GitHub.

