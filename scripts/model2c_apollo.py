import regex as re
import argparse
import torch
from pathlib import Path

TASK = "mnist"
IN_DIMS = {
    "mnist": 784,
}
OUT_DIM = 10
TCM_SIZE = 384 # KB
SRAM_SIZE = 1000 # KB
NVM_SIZE = 2000 # KB
OUT_DIR = Path("models/")
MODEL_DIR = Path("pretrained_models/")

# Lets start easy: dont divide the leaves into neurons and share into memories. keep a whole leaves in a memory (TCM and SRAM). Do that later?
# Lets start easy: keep all nods in one mem. section 

def cal_nodes_size(depth: int) -> float:
    n_nodes = 2 ** depth - 1
    in_dim, out_dim = IN_DIMS[TASK], OUT_DIM
    num_elements = n_nodes * in_dim
    element_size = 4 # Size of each element in bytes (assuming float32)
    total_size_kb = (num_elements * element_size) / 1024  # Convert to KB
    return total_size_kb

def cal_leaf_size(width: int) -> float:
    in_dim, out_dim = IN_DIMS[TASK], OUT_DIM
    num_elements = in_dim * width + width * out_dim
    element_size = 4 # Size of each element in bytes (assuming float32)
    total_size_kb = (num_elements * element_size) / 1024  # Convert to KB
    return total_size_kb

def tensor_to_c_array(tensor):
    """Convert PyTorch tensor to C array string"""
    return str(tensor.flatten().tolist()).replace("[", "{").replace("]", "}").replace("},", "},\n")

def write_config(out_file, depth, leaf_width, n_leaves_in_tcm, remain_leaves):
    # Model Config & Params
    with open(out_file, "w") as f:
        # Config
        f.write(f"#define DEPTH {depth}\n")
        f.write(f"#define LEAF_WIDTH {leaf_width}\n")
        f.write(f"#define N_LEAVES (1 << DEPTH)\n")
        f.write(f"#define N_NODES (N_LEAVES - 1)\n")
        f.write(f"#define N_LEAVES_TCM {n_leaves_in_tcm} \n")
        f.write(f"#define N_LEAVES_SRAM {remain_leaves} \n")

def write_weights(state_dict, out_file, n_leaves_in_tcm):
    # Model Config & Params
    with open(out_file, "w") as f:
        # Params
        nw, nb = state_dict["node_weights"], state_dict["node_biases"]
        lw1, lb1 = state_dict["w1s"], state_dict["b1s"]
        lw2, lb2 = state_dict["w2s"], state_dict["b2s"]
        # TCM
        lw1_1, lb1_1 = lw1[:n_leaves_in_tcm], lb1[:n_leaves_in_tcm]
        lw2_1, lb2_1 = lw2[:n_leaves_in_tcm], lb2[:n_leaves_in_tcm]
        f.write(f"#define NW {tensor_to_c_array(nw)}\n")
        f.write(f"#define NB {tensor_to_c_array(nb)}\n")
        f.write(f"#define LW1_1 {tensor_to_c_array(lw1_1.transpose(1, 2))}\n")
        f.write(f"#define LB1_1 {tensor_to_c_array(lb1_1)}\n")
        f.write(f"#define LW2_1 {tensor_to_c_array(lw2_1.transpose(1, 2))}\n")
        f.write(f"#define LB2_1 {tensor_to_c_array(lb2_1)}\n\n")
        # SRAM
        lw1_2, lb1_2 = lw1[n_leaves_in_tcm:], lb1[n_leaves_in_tcm:]
        lw2_2, lb2_2 = lw2[n_leaves_in_tcm:], lb2[n_leaves_in_tcm:]
        f.write(f"#define LW1_2 {tensor_to_c_array(lw1_2.transpose(1, 2))}\n")
        f.write(f"#define LB1_2 {tensor_to_c_array(lb1_2)}\n")
        f.write(f"#define LW2_2 {tensor_to_c_array(lw2_2.transpose(1, 2))}\n")
        f.write(f"#define LB2_2 {tensor_to_c_array(lb2_2)}\n\n")

def get_config(config_name) -> tuple[int, int]:
    match = re.search(r"_d(\d+)_l(\d+)", config_name)
    if match:
        depth, leaf_width = match.groups()
        return int(depth), int(leaf_width)
    else:
        raise ValueError("Filename does not match expected pattern.")

def main(state_dict_file: str):
    OUT_DIR.mkdir(exist_ok=True)
    depth, leaf_width = get_config(state_dict_file)
    n_leaves = 2 ** depth

    state_dict = torch.load(MODEL_DIR/f"{config_name}.pt", map_location="cpu", 
                            weights_only=True)

    config_filename = f"{config_name}_conf.h"
    weights_filename = f"{config_name}_weights.h"
    out_filename = f"{config_name}.h"

    remain_tcm = TCM_SIZE
    remain_tcm -= cal_nodes_size(depth)
    leaf_size = cal_leaf_size(leaf_width)

    n_leaves_in_tcm = int(remain_tcm // leaf_size)
    if n_leaves_in_tcm == 0:
        raise ValueError("Even a single leaf is too large to fit in TCM.")
    elif n_leaves_in_tcm >= n_leaves:
        raise ValueError(f"All leaves can fit in TCM, no need to use SRAM.")
    remain_leaves = n_leaves - n_leaves_in_tcm
    if remain_leaves * leaf_size > SRAM_SIZE:
        raise ValueError(f"Remaining leaves are too large to fit in SRAM.")

    write_config(OUT_DIR/config_filename, depth, leaf_width, n_leaves_in_tcm, remain_leaves)
    write_weights(state_dict, OUT_DIR/weights_filename, n_leaves_in_tcm)
    with open(OUT_DIR / out_filename, "w") as f:
        f.write(f"#include \"{config_filename}\"\n")
        f.write(f"#include \"{weights_filename}\"")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process file arguments.")
    parser.add_argument('-c', '--config-name', default="mnist_d4_l16" ,help="Configuration name (e.g., 'mnist_d4_l16')")
    config_name = parser.parse_args().config_name
    main(config_name)
