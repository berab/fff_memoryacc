import regex as re
import argparse
import torch
from pathlib import Path

model_sizes = {
    784: "mnist",
}
OUT_DIR = Path("models/")
MODEL_DIR = Path("pretrained_models/")
STATS_DIR = Path("fff_stats/")

def tensor_to_c_array(tensor):
    """Convert PyTorch tensor to C array string"""
    return str(tensor.flatten().tolist()).replace("[", "{").replace("]", "}").replace("},", "},\n")

def write_config(out_file, depth, leaf_width, test_leaf_targets):
    # Model Config & Params
    with open(out_file, "w") as f:
        # Config
        f.write(f"#define DEPTH {depth}\n")
        f.write(f"#define LEAF_WIDTH {leaf_width}\n")
        f.write(f"#define N_LEAVES (1 << DEPTH)\n")
        f.write(f"#define N_NODES (N_LEAVES - 1)\n")
        f.write(f"#define LT {tensor_to_c_array(test_leaf_targets)}\n")

def write_weights(state_dict, out_file):
    # Model Config & Params
    with open(out_file, "w") as f:
        # Params
        nw, nb = state_dict["node_weights"], state_dict["node_biases"]
        lw1, lb1 = state_dict["w1s"], state_dict["b1s"]
        lw2, lb2 = state_dict["w2s"], state_dict["b2s"]
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

def main(state_dict_file: str):
    OUT_DIR.mkdir(exist_ok=True)
    depth, leaf_width = get_config(state_dict_file)

    state_dict = torch.load(MODEL_DIR/f"{config_name}.pt", map_location="cpu", 
                            weights_only=True)
    test_leaf_targets = torch.load(STATS_DIR/f"{config_name}_test_leaves.pt", 
                                  map_location="cpu", weights_only=True)

    config_filename = f"{config_name}_conf.h"
    weights_filename = f"{config_name}_weights.h"
    out_filename = f"{config_name}.h"

    write_config(OUT_DIR/config_filename, depth, leaf_width, test_leaf_targets)
    write_weights(state_dict, OUT_DIR/weights_filename)
    with open(OUT_DIR / out_filename, "w") as f:
        f.write(f"#include \"{config_filename}\"\n")
        f.write(f"#include \"{weights_filename}\"")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process file arguments.")
    parser.add_argument('-c', '--config-name', default="mnist_d4_l16" ,help="Configuration name (e.g., 'mnist_d4_l16')")
    config_name = parser.parse_args().config_name
    main(config_name)
