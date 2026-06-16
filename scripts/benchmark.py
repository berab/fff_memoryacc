import os
import mlflow
from mlflow.tracking import MlflowClient
import subprocess
from urllib.parse import urlparse
import struct
import serial
import time
import argparse

# 1. Configure the experiment details
BAUD_RATE = 9600

def parse_args():
    parser = argparse.ArgumentParser(description="Download MLflow artifacts and run on device.")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port (default: /dev/ttyACM0)")
    parser.add_argument("--dataset", default="mnist", help="Dataset name (default: mnist)")
    parser.add_argument("--device", default="STM", help="Device name (default: STM)")
    parser.add_argument("--target_value", type=float, default=1.0, help="Target parameter value (default: 1.0)")
    parser.add_argument("--target_param", default="entropy_alpha", help="Target parameter name (default: entropy_alpha)")
    parser.add_argument("--mlflow_port", default="8083", help="MLflow tracking port (default: 8083)")
    parser.add_argument("--mode", default=0, type=int, help="Memory mode: SORTED (0), UNSORTED (1), FLASH ONLY (3)")
    return parser.parse_args()

def tensor_bytes_to_c_array(tensor_bytes: bytes) -> str:
    """Deserialize int64 tensor bytes and emit C array initializer."""
    n = len(tensor_bytes) // 8
    values = struct.unpack(f"<{n}q", tensor_bytes)
    body = ", ".join(str(v) for v in values)
    return f"{{ {body} }}"


def build_header(artifacts_dir: str, header_path: str):
    lt_path = os.path.join(artifacts_dir, "test_leaves.pt")
    li_path = os.path.join(artifacts_dir, "val_opt_indices.pt")

    if not (os.path.exists(lt_path) and os.path.exists(li_path)):
        missing = [p for p in (lt_path, li_path) if not os.path.exists(p)]
        print(f"Missing artifact file(s): {missing}")
        return False

    try:
        import torch
    except ImportError:
        print("PyTorch is required to load .pt artifacts.")
        return False

    lt_tensor = torch.load(lt_path, map_location="cpu")
    li_tensor = torch.load(li_path, map_location="cpu")

    def to_bytes(t):
        if hasattr(t, "detach"):
            return t.detach().cpu().numpy().astype("<i8").tobytes()
        return bytes(t)

    lt_bytes = to_bytes(lt_tensor)
    li_bytes = to_bytes(li_tensor)

    lt_arr = tensor_bytes_to_c_array(lt_bytes)
    li_arr = tensor_bytes_to_c_array(li_bytes)

    header = f"""#define LT {lt_arr}
#define LI {li_arr}
"""
    with open(header_path, "w") as f:
        f.write(header)
    print(f"Header written to {header_path}")
    return True

def run_make(mode: int):
    print("Running make clean all...")
    if mode == 0:
        result = subprocess.run(["make", "clean", "all", "SORTED=1"], cwd="..", capture_output=True, text=True)
    elif mode == 1:
        result = subprocess.run(["make", "clean", "all"], cwd="..", capture_output=True, text=True)
    else:
        result = subprocess.run(["make", "clean", "all", "FLASH=1"], cwd="..", capture_output=True, text=True)
    if result.returncode != 0:
        print("Make failed:")
        print(result.stderr)
        return False
    print("Make successful.")
    return True

def flash_device():
    print("Flashing device with OpenOCD...")
    result = subprocess.run(["make", "flash"], cwd="..", capture_output=True, text=True)
    if result.returncode != 0:
        print("Flash failed:")
        print(result.stderr)
        return False
    print("Flash successful.")
    return True

def read_serial(port, timeout=15) -> str:
    print(f"Reading serial output from {port}...")
    latency: str = "Null"
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        start_time = time.time()
        while time.time() - start_time < timeout:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                print(f"Serial: {line}")
                if "elapsed_ms=" in line:
                    latency = line.split("elapsed_ms=")[1]
        ser.close()
    except Exception as e:
        print(f"Serial communication failed: {e}")
    return latency

def main():
    args = parse_args()
    
    experiment_name = f"{args.device}_{args.dataset}_iafff_hard_seed"
    download_dir = f"./downloaded_artifacts/{experiment_name}/{args.device}_{args.dataset}_{args.target_param}_{args.target_value}"
    header_dir = f"../Core/Inc/parameters/{args.dataset}_leafstats.h"
    artifact_subdir = "artifacts"
    header_filename = f"{args.device}_{args.dataset}_{args.target_param}_{args.target_value}"

    # We still use MlflowClient briefly just to search for the run ID based on your criteria
    mlflow_tracking_uri = f"http://localhost:{args.mlflow_port}"
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)

    # Get experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Experiment '{experiment_name}' not found.")
        return

    # Search for the run matching the metric
    filter_string = f"params.{args.target_param} = '{args.target_value}'"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        max_results=5
    )

    if not runs:
        print(f"No runs found matching {args.target_param} = {args.target_value}")
        return

    print(f"Found {len(runs)} matching run(s).")
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 2. Use mlflow.artifacts.download_artifacts to download the files
    for run in runs:
        seed = run.data.params["seed"]
        run_id = run.info.run_id
        artifact_uri = run.info.artifact_uri
        artifact_uri = urlparse(artifact_uri).path

        print(f"\nProcessing Run ID: {run_id} (Seed: {seed})")

        command = [
            "scp", 
            "-r", 
            f"hpc:{artifact_uri}",
            download_dir,
        ]
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"Success! Artifacts successfully downloaded to {download_dir}")
            header_path = os.path.join(download_dir, f"{header_filename}_seed{seed}.h")
            build_header(os.path.join(download_dir, artifact_subdir), header_path)

            print(f"Copying header to {header_dir}...")
            command = [
                "cp", 
                header_path,
                f"{header_dir}",
            ]
            subprocess.run(command, capture_output=True, text=True)

            if run_make(args.mode):
                if flash_device():
                    latency = read_serial(args.port)
            output_file = f"results/time_{args.device}_{args.dataset}.csv"
            with open(output_file, "a") as f:
                f.write(f"{latency},{seed},{args.target_value},{args.mode}\n")
        else:
            print("SCP command failed with error:")
            print(result.stderr)

if __name__ == "__main__":
    main()
