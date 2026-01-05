import matplotlib.pyplot as plt
import numpy as np

# Reproducible node placement
rng = np.random.default_rng(42)
num_nodes = 20
coords = rng.uniform(0, 100, size=(num_nodes, 2))
coords[8] = [50, 50]
coords[1] = [35, 70]
coords[2] = [45, 80]
coords[4] = [70, 60]
coords[17] = [40, 20]

# Predefined route from node 1 to 20 (1-indexed for readability)
route = [1, 9, 8, 18, 17, 20]
# route = [1, 5, 10, 4, 7, 3, 2, 20]
route_indices = [idx - 1 for idx in route]
route_x = coords[route_indices, 0]
route_y = coords[route_indices, 1]

# Gaussian parameters (kept consistent with existing plot)
mu_x, mu_y = 50.0, 50.0
sigma_x, sigma_y = 20.0, 40.0
charging_nodes = {10}

# Grid for contour plot
x_grid = np.linspace(0, 100, 200)
y_grid = np.linspace(0, 100, 200)
X, Y = np.meshgrid(x_grid, y_grid)
Z = np.exp(-0.5 * (((X - mu_x) / sigma_x) ** 2 + ((Y - mu_y) / sigma_y) ** 2))


def gaussian_level(x_vals, y_vals):
    return np.exp(
        -0.5 * (((x_vals - mu_x) / sigma_x) ** 2 + ((y_vals - mu_y) / sigma_y) ** 2)
    )


# Build an expanded route that duplicates charging stops (arrive + depart)
expanded_route = []
for node in route:
    expanded_route.append(node)
    if node in charging_nodes:
        expanded_route.append(node)

expanded_indices = [idx - 1 for idx in expanded_route]
expanded_x = coords[expanded_indices, 0]
expanded_y = coords[expanded_indices, 1]

# Compute SOC along the expanded route based on Gaussian level changes
level_values = gaussian_level(expanded_x, expanded_y)
level_diffs = np.diff(level_values)
segment_distances = np.sqrt(np.diff(expanded_x) ** 2 + np.diff(expanded_y) ** 2)
distance_penalty_factor = 0.1  # SOC drop per unit distance traveled
scale = 60.0 / (np.sum(np.abs(level_diffs)) + 1e-6)  # consume ~60% over route
soc = [100.0]
for i, diff in enumerate(level_diffs):
    if diff > 0:
        diff *= 1.4  # charging gains are halved
    else:
        diff *= 0.7  # discharging losses are doubled
    distance_drop = segment_distances[i] * distance_penalty_factor
    delta_soc = -diff * scale - distance_drop  # positive level_diffs => SOC drop; negative => SOC rise
    next_soc = min(soc[-1] + delta_soc, 100.0)
    soc.append(next_soc)
    # If this step lands on a duplicated charging node, reset to full
    if (
        expanded_route[i] in charging_nodes
        and expanded_route[i] == expanded_route[i + 1]
    ):
        soc[-1] = 100.0

# Create figure with map and SOC subplots
fig, (ax_map, ax_soc) = plt.subplots(
    2, 1, figsize=(9, 12), gridspec_kw={"height_ratios": [2.0, 1.0]}
)

# Contours of the Gaussian field
contours = ax_map.contour(
    X, Y, Z, levels=10, colors="blue", linewidths=0.6, zorder=0
)
ax_map.contourf(
    X, Y, Z, levels=5, cmap="Blues", alpha=0.25, zorder=0, antialiased=True
)

# Draw all edges (complete graph) in light gray
for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        x_values = [coords[i, 0], coords[j, 0]]
        y_values = [coords[i, 1], coords[j, 1]]
        ax_map.plot(x_values, y_values, color="lightgray", linewidth=0.4, zorder=1)

# Draw the chosen route in bold red
ax_map.plot(route_x, route_y, color="red", linewidth=2.5, zorder=3, label="Route 1 → 20")

# Plot nodes (default) and highlight start/end/charging
ax_map.scatter(
    coords[:, 0], coords[:, 1], color="steelblue", edgecolor="white", s=80, zorder=4
)
ax_map.scatter(
    coords[0, 0],
    coords[0, 1],
    color="green",
    edgecolor="white",
    s=110,
    zorder=5,
    label="Start (1)",
)
ax_map.scatter(
    coords[-1, 0],
    coords[-1, 1],
    color="orange",
    edgecolor="white",
    s=110,
    zorder=5,
    label="End (20)",
)
ax_map.scatter(
    coords[9, 0],
    coords[9, 1],
    color="purple",
    marker="s",
    edgecolor="white",
    s=130,
    zorder=6,
    label="Charging (10)",
)

# Label nodes
for idx, (x, y) in enumerate(coords, start=1):
    ax_map.text(
        x + 1.2, y + 1.2, str(idx), fontsize=9, color="midnightblue", zorder=5
    )

ax_map.set_xlim(0, 100)
ax_map.set_ylim(0, 100)
ax_map.set_xlabel("X position")
ax_map.set_ylabel("Y position")
ax_map.set_title("Car route from node 1 to node 20 (random node layout)")
ax_map.legend(loc="upper right")
ax_map.set_aspect("equal", adjustable="box")
ax_map.grid(False)

# SOC subplot
node_positions = np.arange(1, len(expanded_route) + 1)
ax_soc.plot(node_positions, soc, marker="o", color="darkred", linewidth=2)
ax_soc.set_xticks(node_positions)
ax_soc.set_xticklabels([str(n) for n in expanded_route])
ax_soc.set_xlim(0.8, len(expanded_route) + 0.2)
# ax_soc.set_ylim(0, 105)
ax_soc.set_xlabel("Node along route (with charging stops)")
ax_soc.set_ylabel("State of Charge (%)")
ax_soc.set_title("SOC vs route nodes (initial SOC = 100%)")
ax_soc.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

plt.tight_layout()
plt.savefig("car_route_graph.png", dpi=200)
plt.show()
plt.close(fig)
