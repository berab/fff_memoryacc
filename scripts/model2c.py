import regex as re
import argparse
import torch

model_sizes = {
    784: "mnist",
}

def tensor_to_c_array(tensor, name):
    """Convert PyTorch tensor to C array string"""
    array_str = str(tensor.tolist()).replace("[", "{").replace("]", "}")
    array_str = array_str.replace("},", "},\n")  # Format multi-dimensional
    return f"{name} = {array_str};\n\n"

def write_model(state_dict, out_file, task, depth, leaf_width):
    # Model Config & Params
    with open(out_file, "w") as f:
        # Config
        f.write(f"#include \"../bins/{task}.h")
        f.write(f"#define DEPTH {depth}\n")
        f.write(f"#define LEAF_WIDTH {leaf_width}\n")
        f.write(f"#define N_LEAVES (1 << DEPTH)\n")
        f.write(f"#define N_NODES (N_LEAVES - 1)\n")

        # Params
        nw, nb = state_dict["node_weights"], state_dict["node_biases"]
        fc1_w, fc1_b = state_dict["w1s"], state_dict["b1s"]
        fc2_w, fc2_b = state_dict["w2s"], state_dict["b2s"]
        nw = tensor_to_c_array(nw, "float nw[N_NODES][IN_FEATURES]")
        nb = tensor_to_c_array(nb.squeeze(), "float nb[N_NODES]")
        fc1_w = tensor_to_c_array(fc1_w.transpose(1, 2), "float w1[N_LEAVES][LEAF_WIDTH][IN_FEATURES]")
        fc1_b = tensor_to_c_array(fc1_b, "float b1[N_LEAVES][LEAF_WIDTH]")
        fc2_w = tensor_to_c_array(fc2_w.transpose(1, 2), "float w2[N_LEAVES][OUT_FEATURES][LEAF_WIDTH]")
        fc2_b = tensor_to_c_array(fc2_b, "float b2[N_LEAVES][OUT_FEATURES]")
        # DATA
        hidden = "\nfloat hidden[LEAF_WIDTH];"
        for p in [nw, nb, fc1_w, fc1_b, fc2_w, fc2_b, hidden]:
            f.write(p)

def get_config(state_dict_name):
    match = re.search(r"(\w+)_d(\d+)_l(\d+)\.pt", state_dict_name)
    if match:
        task, depth, leaf_width = match.groups()
        return task, depth, leaf_width
    else:
        raise ValueError("Filename does not match expected pattern.")

def main(state_dict_file: str):
    task, depth, leaf_width = get_config(state_dict_file)
    dict_name = state_dict_file.split("/")[-1]
    out_filename = dict_name.replace(".pt", ".h")
    state_dict = torch.load(state_dict_file, map_location="cpu")
    write_model(state_dict, f"Models/{out_filename}", task, depth, leaf_width)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process file arguments.")
    parser.add_argument('-s', '--state_dict', required=True, help='State dictionary file path')
    state_dict_file = parser.parse_args().state_dict
    main(parser.parse_args().state_dict)
