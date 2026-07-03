import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from pathlib import Path


def load_cells(csv_path: str) -> tuple[list, dict]:
    cells_df = pd.read_csv(csv_path)
    cell_list = [
        (row["weather"], row["distance"], row["speed"])
        for _, row in cells_df.iterrows()
    ]
    cell_to_index = {cell: idx for idx, cell in enumerate(cell_list)}
    return cell_list, cell_to_index


def parse_controller_name(controller_path: str) -> str:
    filename = controller_path.split("/")[-1]          # resnet18_fine_best.pt
    stem = filename.rsplit(".", 1)[0]                  # resnet18_fine_best
    parts = stem.split("_")                            # ["resnet18", "fine", "best"]
    # Drop the last token ("best") and rejoin
    return "_".join(parts[:-1])                        # resnet18_fine


def process_history(history, cell_to_index: dict) -> tuple[dict, dict, int]:
    controller_map = {}
    aggregated_rewards = defaultdict(lambda: defaultdict(list))
    skipped_count = 0

    for entry in history:
        if entry["safety_info"].get("safety_violation", 0) == 1:
            skipped_count += 1
            continue

        c_t = entry["c_t"]
        cell = entry["cell"]
        r_t = entry["r_t"]
        controller_name = parse_controller_name(entry["controller_path"])

        if c_t not in controller_map:
            controller_map[c_t] = controller_name
        else:
            if controller_map[c_t] != controller_name:
                print("Mismatch detected!")
                print(f"Existing: {controller_map[c_t]}")
                print(f"New:      {controller_name}")

        if cell in cell_to_index:
            aggregated_rewards[c_t][cell].append(r_t)

    return controller_map, aggregated_rewards, skipped_count


def compute_averages(
    controller_map: dict,
    aggregated_rewards: dict,
    cell_list: list,
    reward_order: str,
) -> dict:
    final_averages = {}

    for c_id in controller_map:
        controller_data = []
        for cell in cell_list:
            rewards = aggregated_rewards[c_id].get(cell, [])
            if rewards:
                avg = np.mean(rewards, axis=0).tolist()
                if reward_order == "se":
                    #print("se changing sequence")
                    avg = [avg[1], avg[0]]  # invert: s_e -> r0=efficiency, r1=stability
            else:
                avg = [0.0, 0.0]
            controller_data.append(avg)
        final_averages[c_id] = controller_data

    return final_averages


def plot_controller_performance(
    target_cell,
    controller_map: dict,
    final_averages: dict,
    cell_to_index: dict,
    output_path: str = "controller_rewards_plot.png",
):
    cell_idx = cell_to_index.get(target_cell)
    if cell_idx is None:
        print(f"Target cell {target_cell} not found.")
        return

    print(f"cell_idx: {cell_idx}")

    plot_data = [
        [controller_map[c_id], *final_averages[c_id][cell_idx]]
        for c_id in controller_map
    ]
    print(plot_data)

    names, r0, r1 = zip(*plot_data)

    plt.figure(figsize=(10, 6))
    plt.scatter(r0, r1, color="royalblue", edgecolors="black", s=100, zorder=3)
    for i, name in enumerate(names):
        plt.text(r0[i], r1[i], f" {name}", fontsize=9, va="center")

    plt.xlabel("Reward Efficiency")
    plt.ylabel("Reward Stability")
    plt.title(f"Controller Performance for Context: {target_cell}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
    print("Graph generated successfully.")


def main():
    # --- Config ---
    #npz_path = "../bandit_checkpoint_e_s.npz"
    npz_path = "../bandit_scalar_lambda_1_0.npz"
    reward_order = "es"
    csv_path = "context_cells.csv"
    extracted_path = "_".join(Path(npz_path).stem.split("_")[-2:])
    output_path = "exp1_plots/"+extracted_path+"_scalarized.png"
    target_cell = ("ClearNoon", 5, 4)

    # --- Load ---
    data = np.load(npz_path, allow_pickle=True)
    cell_list, cell_to_index = load_cells(csv_path)

    # --- Process ---
    controller_map, aggregated_rewards, skipped_count = process_history(
        data["history"], cell_to_index
    )
    final_averages = compute_averages(controller_map, aggregated_rewards, cell_list, reward_order)

    # --- Report ---
    print(f"Total skipped entries (violations): {skipped_count}")
    print("\nController Mapping:")
    for cid, name in controller_map.items():
        print(f"  {cid}: {name}")
    print("\nCell List:")
    for i, cell in enumerate(cell_list):
        print(f"  {i}: {cell}")

    # --- Plot ---
    plot_controller_performance(target_cell, controller_map, final_averages, cell_to_index, output_path)


if __name__ == "__main__":
    main()