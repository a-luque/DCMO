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
    filename = controller_path.split("/")[-1]
    stem = filename.rsplit(".", 1)[0]
    parts = stem.split("_")
    return "_".join(parts[:-1])


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
                    avg = [avg[1], avg[0]]
            else:
                avg = [0.0, 0.0]
            controller_data.append(avg)
        final_averages[c_id] = controller_data

    return final_averages


def load_and_process(npz_path: str, csv_path: str, reward_order: str):
    data = np.load(npz_path, allow_pickle=True)
    cell_list, cell_to_index = load_cells(csv_path)
    controller_map, aggregated_rewards, skipped_count = process_history(
        data["history"], cell_to_index
    )
    final_averages = compute_averages(controller_map, aggregated_rewards, cell_list, reward_order)
    return cell_list, cell_to_index, controller_map, final_averages, skipped_count


def parse_weights_from_label(label: str) -> tuple[float, float] | None:
    """Parse lambda weights from a label like 'lambda_1_0', 'lambda_07_03', etc.
    Returns (w0, w1) as floats, or None if not a scalar lambda label."""
    parts = label.split("_")
    # Expect format: lambda_<w0>_<w1>
    #if len(parts) >= 3 and parts[0] == "lambda":
    if parts[0] != "e":
        def to_float(s):
            # "1" -> 1.0, "0" -> 0.0, "07" -> 0.7, "03" -> 0.3, "05" -> 0.5
            if len(s) == 1:
                return float(s)
            return float(s) / (10 ** (len(s) - 1))
        try:
            return to_float(parts[0]), to_float(parts[1])
        except (ValueError, IndexError):
            return None
    return None


def plot_combined(
    target_cell,
    datasets: list[dict],  # list of {label, controller_map, final_averages, cell_to_index, color, weights (optional)}
    output_path: str = "combined_plot.png",
):
    plt.figure(figsize=(10, 6))

    for ds in datasets:
        cell_idx = ds["cell_to_index"].get(target_cell)
        if cell_idx is None:
            print(f"Target cell {target_cell} not found in dataset '{ds['label']}'.")
            continue

        weights = ds.get("weights")
        #print(f"weights:{weights}")

        if weights is not None:
            # Select only the single best controller by weighted reward
            w0, w1 = weights
            best_c_id = max(
                ds["controller_map"],
                key=lambda c_id: w0 * ds["final_averages"][c_id][cell_idx][0]
                               + w1 * ds["final_averages"][c_id][cell_idx][1],
            )
            best_name = ds["controller_map"][best_c_id]
            r0_val, r1_val = ds["final_averages"][best_c_id][cell_idx]

            plt.scatter(
                [r0_val], [r1_val],
                color=ds["color"],
                edgecolors="black",
                s=100,
                zorder=3,
                label=ds["label"],
            )
            plt.text(r0_val, r1_val, f" {best_name}", fontsize=9, va="center", color=ds["color"])

        else:
            # Plot all controllers (original behaviour)
            plot_data = [
                [ds["controller_map"][c_id], *ds["final_averages"][c_id][cell_idx]]
                for c_id in ds["controller_map"]
            ]

            names, r0, r1 = zip(*plot_data)

            plt.scatter(
                r0, r1,
                color=ds["color"],
                edgecolors="black",
                s=100,
                zorder=3,
                label=ds["label"],
            )
            for i, name in enumerate(names):
                plt.text(r0[i], r1[i], f" {name}", fontsize=9, va="center", color=ds["color"])

    plt.xlabel("Reward Efficiency")
    plt.ylabel("Reward Stability")
    plt.title(f"Controller Performance for Context: {target_cell}")
    #plt.legend(title="Dataset")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
    print("Graph generated successfully.")


def main():
    # --- Config ---
    npz_paths = [
        "../bandit_checkpoint_e_s.npz",
        "../bandit_scalar_lambda_1_0.npz",
        "../bandit_scalar_lambda_0_1.npz",
        "../bandit_scalar_lambda_05_05.npz",
        "../bandit_scalar_lambda_07_03.npz",
        "../bandit_scalar_lambda_03_07.npz",

    ]
    colors = ["royalblue", "tomato", "seagreen", "hotpink", "goldenrod", "sienna"]
    reward_order = "es"
    csv_path = "context_cells.csv"
    target_cell = ("ClearNoon", 5, 4)
    output_path = "exp1_plots/combined_scalarized.png"

    # --- Load & process each dataset ---
    datasets = []
    for npz_path, color in zip(npz_paths, colors):
        label = "_".join(Path(npz_path).stem.split("_")[-2:])
        cell_list, cell_to_index, controller_map, final_averages, skipped_count = load_and_process(
            npz_path, csv_path, reward_order
        )

        print(f"\n[{label}] Skipped entries (violations): {skipped_count}")
        print(f"[{label}] Controller Mapping:")
        for cid, name in controller_map.items():
            print(f"  {cid}: {name}")

        weights = parse_weights_from_label(label)
        legend_label = f"({weights[0]}, {weights[1]})" if weights is not None else "CMO"

        datasets.append({
            "label": legend_label,
            "color": color,
            "cell_to_index": cell_to_index,
            "controller_map": controller_map,
            "final_averages": final_averages,
            "weights": weights,  # None for non-scalar datasets; (w0, w1) for lambda datasets
        })

    # --- Plot ---
    plot_combined(target_cell, datasets, output_path)


if __name__ == "__main__":
    main()