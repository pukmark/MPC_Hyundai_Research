import argparse
import matplotlib.pyplot as plt
import numpy as np

from energy_constrained_dijkstra import (
    find_fastest_feasible_path,
    find_fastest_with_charging,
)


def build_random_graph(n_nodes, charging_nodes=None, seed=0):
    """
    Create a dense random graph with symmetric travel times and SOC deltas.
    Travel times come from Euclidean distance so the triangle inequality holds.
    Energy deltas depend on edge length and squared elevation change (larger
    |dZ| consumes more), with noise and optional charging bonuses.

    Args:
        n_nodes: Number of nodes to generate (>= 2).
        charging_nodes: Optional iterable of charging node indices. If None,
            a subset is chosen randomly (excluding start/end).
        seed: Seed for reproducibility.

    Returns:
        coords: (n, 2) array of node coordinates.
        Tmat: (n, n) travel time matrix (np.inf on diagonal).
        Emat: (n, n) SOC delta matrix (gain if destination is charging).
        charging_nodes: Sorted list of charging node indices.
    """
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2")

    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 100.0, size=(n_nodes, 2))

    Tmat = np.full((n_nodes, n_nodes), np.inf)
    Emat = np.zeros((n_nodes, n_nodes))

    if charging_nodes is None:
        num_chargers = max(1, n_nodes // 20)
        # Exclude start (0) and goal (n_nodes - 1) from random selection
        candidates = list(range(1, n_nodes - 1)) if n_nodes > 2 else []
        rng.shuffle(candidates)
        charging_nodes = sorted(candidates[:num_chargers])
    else:
        charging_nodes = sorted(set(charging_nodes))

    # Parameters to control difficulty of feasibility
    speed = 1.5  # lower is slower (time = dist / speed)
    energy_per_dist = 0.1  # SOC cost per unit distance
    energy_per_dz2 = 1.0  # SOC cost per squared elevation change
    recharge_gain = 35.0
    noise_energy = 2.0

    # Terrain height field for energy calculations (smooth, reproducible)
    def terrain_height(xy):
        x, y = xy
        hill = 6.0 * np.exp(-((x - 5.0) ** 2 + (y - 5.0) ** 2) / 30.0)
        ridge = 3.0 * np.sin(x / 2.0) + 2.0 * np.cos(y / 3.0)
        return hill + ridge

    z_vals = np.array([terrain_height(pt) for pt in coords])

    # Precompute pairwise distances, times, and energies (symmetric).
    dist_mat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    np.fill_diagonal(dist_mat, 0.0)
    time_full = dist_mat / speed  # triangle inequality holds by construction

    energy_full = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            dz = z_vals[j] - z_vals[i]
            energy = (
                -energy_per_dist * dist_mat[i, j]
                - energy_per_dz2 * (dz ** 2)
                + rng.normal(0.0, noise_energy)
            )
            if j in charging_nodes:
                energy += recharge_gain
            if i in charging_nodes:
                energy += recharge_gain
            energy_full[i, j] = energy_full[j, i] = energy

    # For each node, connect to 2 randomly chosen neighbors among its 5 closest.
    for node in range(n_nodes):
        neighbor_order = np.argsort(dist_mat[node])
        nearest = [k for k in neighbor_order if k != node][: min(5, n_nodes - 1)]
        if not nearest:
            continue
        k_select = min(2, len(nearest))
        chosen = rng.choice(nearest, size=k_select, replace=False)
        for neighbor in chosen:
            if not np.isfinite(Tmat[node, neighbor]):
                Tmat[node, neighbor] = Tmat[neighbor, node] = time_full[node, neighbor]
                Emat[node, neighbor] = Emat[neighbor, node] = energy_full[node, neighbor]

    return coords, Tmat, Emat, charging_nodes


def plot_path(coords, T, path, soc_trace, charging_nodes):
    fig, (ax_map, ax_soc) = plt.subplots(
        1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )

    n = len(coords)
    for i in range(n):
        for j in range(i + 1, n):
            if np.isfinite(T[i, j]):
                ax_map.plot(
                    [coords[i, 0], coords[j, 0]],
                    [coords[i, 1], coords[j, 1]],
                    color="lightgray",
                    linewidth=0.9,
                    zorder=1,
                )

    path_coords = coords[path]
    ax_map.plot(
        path_coords[:, 0],
        path_coords[:, 1],
        color="red",
        linewidth=2.7,
        marker="o",
        zorder=4,
        label="Chosen path",
    )

    # Nodes and charging markers
    ax_map.scatter(
        coords[:, 0], coords[:, 1], color="steelblue", edgecolor="white", s=80, zorder=3
    )
    ax_map.scatter(
        path_coords[0, 0],
        path_coords[0, 1],
        color="green",
        edgecolor="white",
        s=110,
        zorder=5,
        label="Start",
    )
    ax_map.scatter(
        path_coords[-1, 0],
        path_coords[-1, 1],
        color="orange",
        edgecolor="white",
        s=110,
        zorder=5,
        label="Goal",
    )
    if charging_nodes:
        ax_map.scatter(
            coords[charging_nodes, 0],
            coords[charging_nodes, 1],
            color="purple",
            marker="s",
            edgecolor="white",
            s=140,
            zorder=6,
            label="Charging",
        )

    for idx, (x, y) in enumerate(coords):
        ax_map.text(x + 0.12, y + 0.12, str(idx), fontsize=9, color="midnightblue")

    ax_map.set_title("Energy-constrained fastest path")
    ax_map.set_xlabel("X")
    ax_map.set_ylabel("Y")
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.legend(loc="upper right")

    steps = np.arange(len(path))
    ax_soc.plot(steps, soc_trace, marker="o", color="darkred", linewidth=2)
    ax_soc.set_xticks(steps)
    ax_soc.set_xticklabels([str(n) for n in path])
    ax_soc.set_xlabel("Node along path")
    ax_soc.set_ylabel("SOC upon arrival")
    ax_soc.set_title("SOC along feasible path")
    ax_soc.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)

    fig.tight_layout()
    plt.savefig("example_energy_path.png", dpi=200)
    plt.show()
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate random Tmat/Emat, solve energy-constrained shortest path, and plot."
    )
    parser.add_argument("--n-nodes", type=int, default=25, help="Number of nodes to generate.")
    parser.add_argument(
        "--start",
        type=int,
        default=18,
        help="Start node index (0-based). Goal defaults to last node.",
    )
    parser.add_argument(
        "--charging-nodes",
        type=int,
        nargs="*",
        help="Explicit charging node indices (0-based). If omitted, they are chosen randomly.",
    )
    parser.add_argument("--soc-init", type=float, default=80.0, help="Initial SOC at start node.")
    parser.add_argument("--soc-min", type=float, default=25.0, help="Minimum SOC allowed.")
    parser.add_argument("--soc-reset", type=float, default=100.0, help="SOC after charging stop.")
    parser.add_argument("--seed", type=int, default=30, help="Random seed for reproducibility.")
    return parser.parse_args()


def main():
    args = parse_args()
    coords, Tmat, Emat, charging_nodes = build_random_graph(
        args.n_nodes,
        charging_nodes=args.charging_nodes,
        seed=args.seed,
    )

    start = args.start
    goal = args.n_nodes - 1
    try:
        path, total_time, soc_trace = find_fastest_with_charging(
            Tmat,
            Emat,
            start=start,
            goal=goal,
            soc_init=args.soc_init,
            soc_min=args.soc_min,
            charging_nodes=charging_nodes,
            soc_reset=args.soc_reset,
        )
        solver = "direct or via one charger (with reset)"
    except ValueError:
        # Fall back to direct feasible path to surface clearer failure message.
        path, total_time, soc_trace = find_fastest_feasible_path(
            Tmat, Emat, start=start, goal=goal, soc_init=args.soc_init, soc_min=args.soc_min
        )
        solver = "direct only"

    print(f"Charging nodes: {charging_nodes}")
    print(f"Solver: {solver}")
    print(f"Path: {path}")
    print(f"Total time: {total_time:.2f}")
    print(f"SOC trace: {[round(s, 1) for s in soc_trace]}")

    plot_path(coords, Tmat, path, soc_trace, charging_nodes)


if __name__ == "__main__":
    main()
