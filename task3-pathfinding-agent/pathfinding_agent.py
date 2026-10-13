"""
Task 3: Heuristic Graph Pathfinding Agent Search Engine
-------------------------------------------------------
Implements A* Search and Dijkstra's algorithm from scratch on 2D grid mazes,
tests them on grids with varying size / obstacle density / maze type, tracks
metrics (steps, cost, runtime, expanded nodes, expanded-node density, frontier size)
and exports visualisation logs (PNG figures, an animated GIF, JSON logs, CSV tables).

Run:      python pathfinding_agent.py
Options:  python pathfinding_agent.py --out results --seeds 10 --diagonal
Needs:    pip install numpy pandas matplotlib
"""
import argparse
import heapq
import json
import math
import os
import time
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")                       # save images without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation

# ====================================================================== 1. GRIDS
# Grid convention: 0 = free cell, 1 = obstacle (wall). Cells are (row, col) tuples.


def random_grid(n, density, seed):
    """n x n grid where each cell is an obstacle with probability `density`."""
    rng = np.random.default_rng(seed)
    grid = (rng.random((n, n)) < density).astype(int)
    grid[0, 0] = 0
    grid[n - 1, n - 1] = 0
    return grid, (0, 0), (n - 1, n - 1)


def solvable_random_grid(n, density, seed, diagonal=False):
    """Like random_grid, but moves on to the next seed until a path from start to goal exists."""
    while True:
        grid, s, g = random_grid(n, density, seed)
        if search(grid, s, g, "dijkstra", diagonal).found:
            return grid, s, g
        seed += 1


def maze_grid(n, seed, loop_fraction=0.08):
    """
    Maze made with a recursive-backtracker (depth-first carving).
    A few extra walls are then removed ("braiding") so that several routes exist;
    otherwise a perfect maze has only one path and no algorithm can be smarter.
    n must be odd.
    """
    if n % 2 == 0:
        n += 1
    rng = np.random.default_rng(seed)
    grid = np.ones((n, n), dtype=int)
    grid[1, 1] = 0
    stack = [(1, 1)]
    while stack:
        r, c = stack[-1]
        options = [(r + dr, c + dc) for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2))
                   if 0 < r + dr < n - 1 and 0 < c + dc < n - 1 and grid[r + dr, c + dc] == 1]
        if not options:
            stack.pop()
            continue
        nr, nc = options[rng.integers(len(options))]
        grid[(r + nr) // 2, (c + nc) // 2] = 0      # knock down the wall between
        grid[nr, nc] = 0
        stack.append((nr, nc))
    # braid: remove some inner walls that separate two free cells
    walls = [(r, c) for r in range(1, n - 1) for c in range(1, n - 1)
             if grid[r, c] == 1 and ((grid[r - 1, c] == 0 and grid[r + 1, c] == 0) or
                                     (grid[r, c - 1] == 0 and grid[r, c + 1] == 0))]
    rng.shuffle(walls)
    for r, c in walls[:int(len(walls) * loop_fraction)]:
        grid[r, c] = 0
    return grid, (1, 1), (n - 2, n - 2)


# =================================================================== 2. SEARCH
@dataclass
class SearchResult:
    algorithm: str
    found: bool
    path: list = field(default_factory=list)
    cost: float = math.inf
    steps: int = 0                  # number of moves along the path
    expanded: int = 0               # nodes removed from the frontier and processed
    generated: int = 0              # nodes pushed to the frontier
    max_frontier: int = 0           # peak size of the open list (memory use)
    runtime_ms: float = 0.0
    free_cells: int = 0
    expanded_order: list = field(default_factory=list)

    @property
    def expanded_density(self):     # share of all free cells the search had to explore
        return self.expanded / self.free_cells if self.free_cells else 0.0


def neighbours(grid, node, diagonal):
    """Valid moves from `node` with their cost. Diagonal moves can't cut wall corners."""
    r, c = node
    rows, cols = grid.shape
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        moves += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dr, dc in moves:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < rows and 0 <= nc < cols) or grid[nr, nc] == 1:
            continue
        if dr != 0 and dc != 0:                       # diagonal: both side cells must be free
            if grid[r + dr, c] == 1 or grid[r, c + dc] == 1:
                continue
            yield (nr, nc), math.sqrt(2)
        else:
            yield (nr, nc), 1.0


def heuristic(a, b, diagonal):
    """Admissible heuristic: never overestimates the true remaining cost."""
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    if diagonal:                                      # octile distance
        return (dr + dc) + (math.sqrt(2) - 2) * min(dr, dc)
    return dr + dc                                    # Manhattan distance


def search(grid, start, goal, algorithm="astar", diagonal=False):
    """
    Best-first graph search.
      Dijkstra : priority = g(n)            (cost so far)           -> heuristic is 0
      A*       : priority = g(n) + h(n)     (cost so far + guess)
    Uses a binary heap (heapq) as the priority queue and lazy deletion of stale entries.
    """
    use_h = algorithm == "astar"
    t0 = time.perf_counter()

    g_cost = {start: 0.0}
    parent = {start: None}
    closed = set()
    counter = 0                                       # tie-breaker so heap never compares nodes
    h0 = heuristic(start, goal, diagonal) if use_h else 0.0
    frontier = [(h0, h0, counter, start)]             # (f, h, tie, node)
    generated, max_frontier = 1, 1
    order = []

    found = False
    while frontier:
        f, h, _, node = heapq.heappop(frontier)
        if node in closed:                            # stale duplicate entry
            continue
        closed.add(node)
        order.append(node)
        if node == goal:
            found = True
            break
        for nxt, step_cost in neighbours(grid, node, diagonal):
            if nxt in closed:
                continue
            new_g = g_cost[node] + step_cost
            if new_g < g_cost.get(nxt, math.inf):
                g_cost[nxt] = new_g
                parent[nxt] = node
                hn = heuristic(nxt, goal, diagonal) if use_h else 0.0
                counter += 1
                heapq.heappush(frontier, (new_g + hn, hn, counter, nxt))
                generated += 1
        max_frontier = max(max_frontier, len(frontier))

    runtime_ms = (time.perf_counter() - t0) * 1000
    path = []
    if found:                                         # walk back through parents
        node = goal
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()

    return SearchResult(
        algorithm="A*" if use_h else "Dijkstra", found=found, path=path,
        cost=g_cost[goal] if found else math.inf, steps=max(len(path) - 1, 0),
        expanded=len(order), generated=generated, max_frontier=max_frontier,
        runtime_ms=runtime_ms, free_cells=int((grid == 0).sum()), expanded_order=order)


# ============================================================= 3. VISUALISATION
COLORS = {"free": (1, 1, 1), "wall": (0.15, 0.15, 0.18), "explored": (0.68, 0.84, 0.95)}


def render(grid, start, goal, explored=None, path=None):
    img = np.empty(grid.shape + (3,))
    img[:] = COLORS["free"]
    img[grid == 1] = COLORS["wall"]
    for r, c in (explored or []):
        img[r, c] = COLORS["explored"]
    return img


def draw_panel(ax, grid, start, goal, result, explored=None, title=None):
    explored = result.expanded_order if explored is None else explored
    ax.imshow(render(grid, start, goal, explored), interpolation="nearest")
    if result.found:
        ys, xs = zip(*result.path)
        ax.plot(xs, ys, color="crimson", linewidth=2)
    ax.scatter([start[1]], [start[0]], c="limegreen", s=60, zorder=3, edgecolors="black")
    ax.scatter([goal[1]], [goal[0]], c="gold", s=80, marker="*", zorder=3, edgecolors="black")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title or f"{result.algorithm}", fontsize=10)


def save_comparison(grid, start, goal, results, title, filename):
    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 5.6))
    axes = np.atleast_1d(axes)
    for ax, res in zip(axes, results):
        label = (f"{res.algorithm}\nsteps={res.steps}  cost={res.cost:.2f}  "
                 f"expanded={res.expanded} ({res.expanded_density:.0%} of free cells)  "
                 f"time={res.runtime_ms:.1f} ms") if res.found else f"{res.algorithm}\nNO PATH"
        draw_panel(ax, grid, start, goal, res, title=label)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(filename, dpi=130)
    plt.close(fig)


def save_animation(grid, start, goal, results, filename, frames=45):
    """GIF that replays the order in which each algorithm explores the grid."""
    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 5.4))
    axes = np.atleast_1d(axes)
    longest = max(len(r.expanded_order) for r in results)
    step = max(1, math.ceil(longest / frames))

    def update(i):
        for ax, res in zip(axes, results):
            ax.clear()
            upto = min((i + 1) * step, len(res.expanded_order))
            shown = res.expanded_order[:upto]
            done = upto >= len(res.expanded_order)
            ax.imshow(render(grid, start, goal, shown), interpolation="nearest")
            if done and res.found:
                ys, xs = zip(*res.path)
                ax.plot(xs, ys, color="crimson", linewidth=2)
            ax.scatter([start[1]], [start[0]], c="limegreen", s=50, edgecolors="black")
            ax.scatter([goal[1]], [goal[0]], c="gold", s=70, marker="*", edgecolors="black")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{res.algorithm}: explored {upto}/{res.expanded}", fontsize=10)

    anim = animation.FuncAnimation(fig, update, frames=math.ceil(longest / step) + 8, interval=120)
    anim.save(filename, writer=animation.PillowWriter(fps=8))
    plt.close(fig)


def save_json_log(grid, start, goal, results, filename, meta):
    """Machine-readable log: grid, metrics, exact path and exploration order per algorithm."""
    log = {"meta": meta, "start": list(start), "goal": list(goal), "grid": grid.tolist(), "runs": []}
    for r in results:
        log["runs"].append({
            "algorithm": r.algorithm, "found": r.found, "cost": None if not r.found else round(r.cost, 4),
            "steps": r.steps, "expanded": r.expanded, "generated": r.generated,
            "max_frontier": r.max_frontier, "runtime_ms": round(r.runtime_ms, 4),
            "expanded_density": round(r.expanded_density, 4),
            "path": [list(p) for p in r.path],
            "expanded_order": [list(p) for p in r.expanded_order]})
    with open(filename, "w") as f:
        json.dump(log, f)


# ================================================================ 4. EXPERIMENTS
def run_experiments(sizes, densities, seeds, maze_sizes, diagonal):
    rows = []
    for n in sizes:
        for d in densities:
            for seed in range(seeds):
                grid, s, g = random_grid(n, d, seed)
                rows += evaluate(grid, s, g, diagonal, {"type": "random", "size": n, "obstacle_density": d, "seed": seed})
    for n in maze_sizes:
        for seed in range(seeds):
            grid, s, g = maze_grid(n, seed)
            rows += evaluate(grid, s, g, diagonal, {"type": "maze", "size": grid.shape[0], "obstacle_density": round(float(grid.mean()), 2), "seed": seed})
    return pd.DataFrame(rows)


def evaluate(grid, s, g, diagonal, meta):
    dij = search(grid, s, g, "dijkstra", diagonal)
    ast = search(grid, s, g, "astar", diagonal)
    if dij.found != ast.found or (dij.found and abs(dij.cost - ast.cost) > 1e-9):
        raise AssertionError(f"A* and Dijkstra disagree on {meta}")     # optimality sanity check
    out = []
    for r in (dij, ast):
        out.append({**meta, "algorithm": r.algorithm, "solved": r.found, "steps": r.steps,
                    "cost": round(r.cost, 3) if r.found else None, "expanded": r.expanded,
                    "expanded_density_pct": round(100 * r.expanded_density, 2),
                    "max_frontier": r.max_frontier, "runtime_ms": round(r.runtime_ms, 3)})
    return out


def plot_summary(df, filename):
    solved = df[df["solved"]]
    rnd = solved[solved["type"] == "random"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    colors = {"Dijkstra": "tab:blue", "A*": "tab:red"}

    for algo, grp in rnd.groupby("algorithm"):
        m = grp.groupby("obstacle_density")["expanded_density_pct"].mean()
        axes[0].plot(m.index * 100, m.values, marker="o", label=algo, color=colors[algo])
        m = grp.groupby("size")["runtime_ms"].mean()
        axes[1].plot(m.index, m.values, marker="o", label=algo, color=colors[algo])
    axes[0].set(title="Exploration effort vs obstacle density", xlabel="Obstacle density (%)",
                ylabel="Expanded nodes (% of free cells)")
    axes[1].set(title="Runtime vs grid size", xlabel="Grid size (n x n)", ylabel="Mean runtime (ms)")

    mz = solved[solved["type"] == "maze"].groupby(["size", "algorithm"])["expanded"].mean().unstack()
    mz.plot(kind="bar", ax=axes[2], color=[colors[c] for c in mz.columns], rot=0)
    axes[2].set(title="Mazes: mean expanded nodes", xlabel="Maze size", ylabel="Expanded nodes")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=130)
    plt.close(fig)


# ======================================================================= 5. MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results", help="output folder")
    ap.add_argument("--seeds", type=int, default=10, help="random grids per setting")
    ap.add_argument("--diagonal", action="store_true", help="allow 8-direction movement")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)
    mode = "8-direction (diagonal)" if args.diagonal else "4-direction"
    print(f"Movement mode: {mode}\n")

    # ---- A. batch experiments over many grids -------------------------------------
    df = run_experiments(sizes=[20, 40, 60], densities=[0.10, 0.20, 0.30, 0.40],
                         seeds=args.seeds, maze_sizes=[21, 41, 61], diagonal=args.diagonal)
    df.to_csv(os.path.join(args.out, "experiments.csv"), index=False)

    solved = df[df["solved"]]
    summary = (solved.groupby(["type", "size", "obstacle_density", "algorithm"])
               [["steps", "expanded", "expanded_density_pct", "max_frontier", "runtime_ms"]]
               .mean().round(2).reset_index())
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    unsolvable = df[(~df["solved"]) & (df["algorithm"] == "A*")]
    print(f"Total runs: {len(df)}  |  unsolvable grids (no path exists): {len(unsolvable)}")

    overall = solved.groupby("algorithm")[["expanded", "expanded_density_pct", "runtime_ms", "max_frontier"]].mean().round(2)
    print("\nAverage over all solved grids:\n", overall.to_string())
    saving = 100 * (1 - overall.loc["A*", "expanded"] / overall.loc["Dijkstra", "expanded"])
    print(f"\nA* expanded {saving:.1f}% fewer nodes than Dijkstra and found paths of identical (optimal) cost.\n")
    plot_summary(df, os.path.join(args.out, "figures", "summary_comparison.png"))

    # ---- B. detailed demo cases with visual logs ----------------------------------
    demos = [("random_open", *solvable_random_grid(40, 0.15, 3, args.diagonal), "Random grid 40x40, 15% obstacles"),
             ("random_dense", *solvable_random_grid(40, 0.30, 7, args.diagonal), "Random grid 40x40, 30% obstacles"),
             ("maze", *maze_grid(41, 5), "Braided maze 41x41")]
    for name, grid, s, g, title in demos:
        results = [search(grid, s, g, "dijkstra", args.diagonal), search(grid, s, g, "astar", args.diagonal)]
        save_comparison(grid, s, g, results, title, os.path.join(args.out, "figures", f"path_{name}.png"))
        save_json_log(grid, s, g, results, os.path.join(args.out, "logs", f"{name}.json"),
                      {"title": title, "movement": mode})
        if name == "maze":
            save_animation(grid, s, g, results, os.path.join(args.out, "figures", "search_animation_maze.gif"))
        for r in results:
            print(f"[{title}] {r.algorithm:8s} found={r.found}  steps={r.steps:3d}  cost={r.cost:7.2f}  "
                  f"expanded={r.expanded:4d} ({r.expanded_density:5.1%})  time={r.runtime_ms:6.2f} ms")
    print(f"\nAll results saved in the '{args.out}' folder.")


if __name__ == "__main__":
    main()
