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



# Create figure with map and SOC subplots
fig = plt.figure(figsize=(10, 12))
ax_map = fig.add_subplot(1, 1, 1)

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

# plt.tight_layout()
plt.savefig("map_graph.png", dpi=200)
plt.show()
plt.close(fig)
