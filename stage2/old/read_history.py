import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

file_path = "../bandit_checkpoint_e_s.npz"
data = np.load(file_path, allow_pickle=True)
history = data["history"]

# -----------------------------
# Initialize containers
# -----------------------------
controller_map = {}
aggregated_rewards = defaultdict(lambda: defaultdict(list))
skipped_count = 0
unique_cells = set()

# -----------------------------
# Process history
# -----------------------------
for entry in history:
    # Skip safety violations
    if entry["safety_info"].get("safety_violation", 0) == 1:
        skipped_count += 1
        continue

    c_t = entry["c_t"]
    cell = entry["cell"]
    r_t = entry["r_t"]
    controller_path = entry["controller_path"]
    controller_name = controller_path.split("/")[-1]

    # Build controller map
    if c_t not in controller_map:
        controller_map[c_t] = controller_name
    else:
        if controller_map[c_t] != controller_name:
            print("Mismatch detected!")
            print(f"Existing: {controller_map[c_t]}")
            print(f"New:      {controller_name}")

    # Store context and reward
    unique_cells.add(cell)
    aggregated_rewards[c_t][cell].append(r_t)

# -----------------------------
# Create stable cell indexing
# -----------------------------
cell_list = sorted(list(unique_cells))
cell_to_index = {cell: idx for idx, cell in enumerate(cell_list)}

# -----------------------------
# Compute average rewards
# -----------------------------
final_averages = {}

for c_id in controller_map.keys():
    controller_data = []

    for cell in cell_list:
        rewards = aggregated_rewards[c_id].get(cell, [])

        if rewards:
            avg = np.mean(rewards, axis=0).tolist()
        else:
            avg = [0.0, 0.0]

        controller_data.append(avg)

    final_averages[c_id] = controller_data

# -----------------------------
# Outputs
# -----------------------------
print(f"Total skipped entries (violations): {skipped_count}")

print("\nController Mapping:")
for cid, name in controller_map.items():
    print(f"{cid}: {name}")

print("\nCell List:")
for i, cell in enumerate(cell_list):
    print(f"{i}: {cell}")

# -----------------------------
# Plot target cell
# -----------------------------
target_cell = ('HardRainNoon', 100, 0)

# FIXED: remove quotes around target_cell
cell_idx = cell_to_index.get(target_cell, None)

if cell_idx is None:
    print(f"Target cell {target_cell} not found.")
else:
    print(f"cell_idx: {cell_idx}")

    plot_data = []

    for c_id in controller_map.keys():
        controller_data = final_averages[c_id]
        rwd = controller_data[cell_idx]

        # FIXED: c_id instead of cid
        plot_data.append([controller_map[c_id], rwd[0], rwd[1]])

    print(plot_data)

    # -----------------------------
    # Scatter plot
    # -----------------------------
    names, r0, r1 = zip(*plot_data)

    plt.figure(figsize=(10, 6))
    plt.scatter(r0, r1, color="royalblue", edgecolors="black", s=100, zorder=3)

    for i, name in enumerate(names):
        plt.text(r0[i], r1[i], f" {name}", fontsize=9, va="center")

    plt.xlabel("Reward Efficiency ($r_0$)")
    plt.ylabel("Reward Stability ($r_1$)")
    plt.title(f"Controller Performance for Context: {target_cell}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    plt.savefig("controller_rewards_plot.png", dpi=300)
    plt.show()

    print("Graph generated successfully.")