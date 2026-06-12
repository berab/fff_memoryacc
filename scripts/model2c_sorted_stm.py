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
TASK = "mnist"
IN_DIMS = {"mnist": 784}
OUT_DIM = 10
SRAM_SIZE = 94
FLASH_SIZE = 1000

def cal_nodes_size(depth: int) -> float:
    n_nodes = 2 ** depth - 1
    in_dim = IN_DIMS[TASK]
    num_elements = n_nodes * in_dim
    element_size = 4
    total_size_kb = (num_elements * element_size) / 1024
    return total_size_kb

def cal_leaf_size(width: int) -> float:
    in_dim, out_dim = IN_DIMS[TASK], OUT_DIM
    num_elements = in_dim * width + width * out_dim
    element_size = 4
    total_size_kb = (num_elements * element_size) / 1024
    return total_size_kb

def get_leaf_stats(leaves, n_leaves) -> list[float]:
    stats = [leaves.count(i)/len(leaves) for i in range(n_leaves)]
    return stats

def tensor_to_c_array(tensor: torch.Tensor):
    """Convert PyTorch tensor to C array string"""
    return str(tensor.flatten().tolist()).replace("[", "{").replace("]", "}").replace("},", "},\n")

def write_config(out_file, depth, leaf_width, n_leaves_in_sram, leaves, new_leaf_order):
    # Model Config & Params
    with open(out_file, "w") as f:
        # Config
        f.write(f"#define DEPTH {depth}\n")
        f.write(f"#define LEAF_WIDTH {leaf_width}\n")
        f.write(f"#define N_LEAVES (1 << DEPTH)\n")
        f.write(f"#define N_NODES (N_LEAVES - 1)\n")
        f.write(f"#define N_LEAVES_SRAM {n_leaves_in_sram} \n")
        f.write(f"#define LT {tensor_to_c_array(leaves)}\n")
        f.write(f"#define LI {tensor_to_c_array(new_leaf_order)}\n")

def write_weights_sorted(state_dict, out_file, sorted_indices):
    # Model Config & Params
    with open(out_file, "w") as f:
        # Params
        f.write(f"#define NW {{0}}\n")
        f.write(f"#define NB {{0}}\n")
        f.write(f"#define LW1_1 {{0}}\n")
        f.write(f"#define LB1_1 {{0}}\n")
        f.write(f"#define LW2_1 {{0}}\n")
        f.write(f"#define LB2_1 {{0}}\n\n")
        f.write(f"#define LW1_2 {{0}}\n")
        f.write(f"#define LB1_2 {{0}}\n")
        f.write(f"#define LW2_2 {{0}}\n")
        f.write(f"#define LB2_2 {{0}}\n\n")

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
    sram, flash = leaf_stats[:6].sum(), leaf_stats[6:].sum()
    sram_sorted, flash_sorted = leaf_stats_sorted[:6].sum(), leaf_stats_sorted[6:].sum()
    print(f"new leaf order: {new_leaf_order}")
    print(f"new leaf order?: {leaf_indices_sorted}")
    print(f"usage | SRAM: {sram}, FLASH: {flash}")
    print(f"usage sortred | SRAM: {sram_sorted}, FLASH: {flash_sorted}")
    return leaf_stats, leaf_indices_sorted, new_leaf_order

def main(config_name):
    OUT_DIR.mkdir(exist_ok=True)
    depth, leaf_width = get_config(config_name)
    n_leaves = 2 ** int(depth)

    state_dict = torch.load(MODEL_DIR/f"{config_name}.pt", map_location="cpu",
                            weights_only=True)
    new_leaf_order = torch.load(STATS_DIR/f"{config_name}_val_opt_indices.pt",
                                map_location="cpu", weights_only=True)
    test_leaves = torch.load(STATS_DIR/f"{config_name}_test_leaves.pt",
                             map_location="cpu", weights_only=True)

    config_filename = f"{config_name}_conf.h"
    weights_filename = f"{config_name}_weights.h"
    out_filename = f"{config_name}.h"

    remain_sram = SRAM_SIZE
    remain_sram -= cal_nodes_size(int(depth))
    leaf_size = cal_leaf_size(int(leaf_width))

    n_leaves_in_sram = int(remain_sram // leaf_size)
    if n_leaves_in_sram == 0:
        raise ValueError("No leaves fit in SRAM.")
    elif n_leaves_in_sram >= n_leaves:
        raise ValueError(f"All leaves can fit in SRAM, no need to use FLASH.")
    remain_leaves = n_leaves - n_leaves_in_sram
    if remain_leaves * leaf_size > FLASH_SIZE:
        raise ValueError(f"Remaining leaves are too large to fit in FLASH.")

    leaves_t, _ = test_leaves.sort()
    write_config(OUT_DIR/config_filename, int(depth), int(leaf_width), n_leaves_in_sram, leaves_t, new_leaf_order)
    sorted_indices = new_leaf_order
    write_weights_sorted(state_dict, OUT_DIR/weights_filename, sorted_indices)
    with open(OUT_DIR / out_filename, "w") as f:
        f.write(f"#include \"{config_filename}\"\n")
        f.write(f"#include \"{weights_filename}\"")

    leaf_stats_df = pd.DataFrame({"usage": test_leaves.tolist()})
    leaf_stats_df.index.name = "leaf"
    leaf_stats_df.to_csv("val_leaf_stats.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process file arguments.")
    parser.add_argument('-c', '--config-name', default="mnist_a08_h05_d4_l4", 
                        help="Configuration name (e.g., 'mnist_d4_l16')")
    config_name = parser.parse_args().config_name
    main(config_name)
