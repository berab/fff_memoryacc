import regex as re
import argparse
import torch
from pathlib import Path
import pandas as pd

model_sizes = {
    784: "mnist",
}
OUT_DIR = Path("models/")
MODEL_DIR = Path("pretrained_models/")
STATS_DIR = Path("fff_stats/")

def get_leaf_stats(leaves, n_leaves) -> list[float]:
    stats = [leaves.count(i)/len(leaves) for i in range(n_leaves)]
    return stats

def tensor_to_c_array(tensor: torch.Tensor):
    """Convert PyTorch tensor to C array string"""
    return str(tensor.flatten().tolist()).replace("[", "{").replace("]", "}").replace("},", "},\n")

def write_config(out_file, depth, leaf_width, leaves: torch.Tensor, new_leaf_order: torch.Tensor):
    # Model Config & Params
    with open(out_file, "w") as f:
        # Config
        f.write(f"#define DEPTH {depth}\n")
        f.write(f"#define LEAF_WIDTH {leaf_width}\n")
        f.write(f"#define N_LEAVES (1 << DEPTH)\n")
        f.write(f"#define N_NODES (N_LEAVES - 1)\n")
        f.write(f"#define LT {tensor_to_c_array(leaves)}\n")
        f.write(f"#define LI {tensor_to_c_array(new_leaf_order)}\n")

def write_weights_sorted(state_dict, out_file, sorted_indices):
    # Model Config & Params
    with open(out_file, "w") as f:
        # Params
        nw, nb = state_dict["node_weights"], state_dict["node_biases"]
        lw1, lb1 = state_dict["w1s"][sorted_indices], state_dict["b1s"][sorted_indices]
        lw2, lb2 = state_dict["w2s"][sorted_indices], state_dict["b2s"][sorted_indices]
        f.write(f"#define NW {tensor_to_c_array(nw)}\n")
        f.write(f"#define NB {tensor_to_c_array(nb)}\n")
        f.write(f"#define LW1 {tensor_to_c_array(lw1.transpose(1, 2))}\n")
        f.write(f"#define LB1 {tensor_to_c_array(lb1)}\n")
        f.write(f"#define LW2 {tensor_to_c_array(lw2.transpose(1, 2))}\n")
        f.write(f"#define LB2 {tensor_to_c_array(lb2)}\n\n")

def get_config(config_name):
    match = re.search(r"_d(\d+)_l(\d+)", config_name)
    if match:
        depth, leaf_width = match.groups()
        return depth, leaf_width
    else:
        raise ValueError("Filename does not match expected pattern.")

def get_new_leaf_order(leaves: list[int], depth: int):
    n_leaves = 2 ** depth
    leaf_stats = torch.tensor(get_leaf_stats(leaves, n_leaves))
    leaf_stats_sorted, leaf_indices_sorted = torch.sort(leaf_stats, descending=True)
    new_leaf_order = torch.empty_like(leaf_indices_sorted)
    new_leaf_order[leaf_indices_sorted] = torch.arange(n_leaves)
    print(f"Val leaf stats: {leaf_stats}")
    print(f"Val leaf stats sorted: {leaf_stats_sorted}")
    tcm, ram = leaf_stats[:6].sum(), leaf_stats[6:].sum()
    tcm_sorted, ram_sorted = leaf_stats_sorted[:6].sum(), leaf_stats_sorted[6:].sum()
    print(f"new leaf order: {new_leaf_order}")
    print(f"new leaf order?: {leaf_indices_sorted}")
    print(f"usage | TCM: {tcm}, RAM: {ram}")
    print(f"usage sortred | TCM: {tcm_sorted}, RAM: {ram_sorted}")
    return leaf_stats, leaf_indices_sorted, new_leaf_order

def main(config_name):
    OUT_DIR.mkdir(exist_ok=True)
    depth, leaf_width = get_config(config_name)

    state_dict = torch.load(MODEL_DIR/f"{config_name}.pt", map_location="cpu", 
                            weights_only=True)
    leaves: list[int] = torch.load(STATS_DIR/f"{config_name}_leaves.pt", map_location="cpu", 
                                     weights_only=True)
    leaf_stats, sorted_indices, new_leaf_order = get_new_leaf_order(leaves, int(depth))

    config_filename = f"{config_name}_conf_sorted.h"
    weights_filename = f"{config_name}_weights_sorted.h"
    out_filename = f"{config_name}_sorted.h"

    leaves_t, _ = torch.tensor(leaves).sort()
    write_config(OUT_DIR/config_filename, depth, leaf_width, leaves_t, new_leaf_order)
    write_weights_sorted(state_dict, OUT_DIR/weights_filename, sorted_indices)
    with open(OUT_DIR / out_filename, "w") as f:
        f.write(f"#include \"{config_filename}\"\n")
        f.write(f"#include \"{weights_filename}\"")

    leaf_stats_df = pd.DataFrame({"usage": leaf_stats.tolist()})
    leaf_stats_df.index.name = "leaf"
    leaf_stats_df.to_csv("val_leaf_stats.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process file arguments.")
    parser.add_argument('-c', '--config-name', default="mnist_d4_l16", 
                        help="Configuration name (e.g., 'mnist_d4_l16')")
    config_name = parser.parse_args().config_name
    main(config_name)
