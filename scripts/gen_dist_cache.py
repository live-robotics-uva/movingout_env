"""Generate BFS distance caches for walled maps (fast flood-fill version).

The repo's original pairwise generator (planner/cache_distance.py) runs one BFS
per (start, end) pair (5.7M pairs) and used a 48-point linspace grid that does
NOT match the runtime lookup indexing (round((x+1.2)/0.05) on a 49x49 grid).
This generator instead runs one multi-target flood fill per source cell on the
SAME 49x49 grid the env uses at runtime (discretize_environment + bfs.py
indexing), so distances agree exactly with live BFS.

Output per map: <json_path>.distcache.npz with
  all_pairs: (2401, 2401) uint16, #moves between cells (65535 = unreachable)
  to_goal:   (2401,)       uint16, #moves to the nearest goal-region cell
Distance in world units = value * 0.05 (matches len(path) * 0.05 in the env).

Usage: python gen_dist_cache.py <map1.json> [map2.json ...] [--workers N]
"""
import json
import multiprocessing as mp
import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moving_out.planner import discrete_env  # noqa: E402

RES = 0.05
NEIGH = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
UNREACH = np.uint16(65535)

_grid = None


def _init(g):
    global _grid
    _grid = g


def _flood(src_flat):
    """BFS flood from one source cell; returns (src_flat, dist row uint16)."""
    H, W = _grid.shape
    si, sj = divmod(src_flat, W)
    dist = np.full(H * W, UNREACH, dtype=np.uint16)
    if _grid[si][sj] == 1:
        return src_flat, dist
    dist[src_flat] = 0
    q = deque([(si, sj)])
    while q:
        ci, cj = q.popleft()
        d = dist[ci * W + cj] + 1
        for dx, dy in NEIGH:
            ni, nj = ci + dx, cj + dy
            if 0 <= ni < H and 0 <= nj < W and _grid[ni][nj] != 1:
                f = ni * W + nj
                if dist[f] == UNREACH:
                    dist[f] = d
                    q.append((ni, nj))
    return src_flat, dist


def build_cache(map_path, workers):
    with open(map_path) as f:
        data = json.load(f)
    data = data[0] if isinstance(data, list) else data
    walls = [w for w in data["walls"] if w and w[0] is not None]
    goal_region = data["target_areas"]
    X = np.array([(-1.2, -1.2), (1.2, 1.2)])
    grid = discrete_env.discretize_environment(X, np.array(walls), goal_region, RES)
    H, W = grid.shape
    n = H * W
    print(f"{os.path.basename(map_path)}: grid {H}x{W}, "
          f"obstacles={int((grid == 1).sum())}, goal_cells={int((grid == 2).sum())}",
          flush=True)

    all_pairs = np.full((n, n), UNREACH, dtype=np.uint16)
    with mp.Pool(workers, initializer=_init, initargs=(grid,)) as pool:
        for src, row in pool.imap_unordered(_flood, range(n), chunksize=64):
            all_pairs[src] = row

    # multi-source BFS from all goal cells
    to_goal = np.full(n, UNREACH, dtype=np.uint16)
    q = deque()
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 2:
                to_goal[i * W + j] = 0
                q.append((i, j))
    while q:
        ci, cj = q.popleft()
        d = to_goal[ci * W + cj] + 1
        for dx, dy in NEIGH:
            ni, nj = ci + dx, cj + dy
            if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] != 1:
                f = ni * W + nj
                if to_goal[f] == UNREACH:
                    to_goal[f] = np.uint16(d)
                    q.append((ni, nj))

    out = map_path + ".distcache.npz"
    np.savez_compressed(out, all_pairs=all_pairs, to_goal=to_goal,
                        shape=np.array([H, W]), resolution=np.array([RES]))
    print(f"  -> {out} ({os.path.getsize(out)/1e6:.1f} MB)", flush=True)


def _resolve_maps(names):
    """Map names (AVAILABLE_MAPS keys) or json paths -> json paths.
    Without names: every map that has walls (wall-free maps use euclidean
    distance and need no cache)."""
    from moving_out.env_parameters import AVAILABLE_MAPS, DEFAULT_MAP_PATH

    if names:
        paths = []
        for n in names:
            if os.path.exists(n):
                paths.append(n)
            elif n in AVAILABLE_MAPS:
                paths.append(os.path.join(DEFAULT_MAP_PATH, AVAILABLE_MAPS[n]))
            else:
                sys.exit(f"unknown map {n!r} (not a file or AVAILABLE_MAPS key)")
        return paths
    paths = []
    for key, json_name in AVAILABLE_MAPS.items():
        if not isinstance(key, str):
            continue  # numeric aliases duplicate the string keys
        p = os.path.join(DEFAULT_MAP_PATH, json_name)
        try:
            walls = json.load(open(p)).get("walls") or []
        except FileNotFoundError:
            continue
        if any(w and w[0] is not None for w in walls):
            paths.append(p)
    return paths


if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    workers = 30
    for a in sys.argv[1:]:
        if a.startswith("--workers"):
            workers = int(a.split("=")[1])
    for mp_path in _resolve_maps(names):
        build_cache(mp_path, workers)
