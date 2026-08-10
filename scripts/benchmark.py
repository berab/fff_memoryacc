import os
import mlflow
from mlflow.tracking import MlflowClient
import subprocess
from urllib.parse import urlparse
import struct
import serial
import time
import argparse
from pathlib import Path

# 1. Configure the experiment details
BAUD_RATE = 9600
SERVERS = {
    "8081": "ws1",
    "8082": "ws2",
    "8083": "hpc",
    "8084": "baldo",
}

def parse_args():
    parser = argparse.ArgumentParser(description="Download MLflow artifacts and run on device.")
    parser.add_argument("--dbg", default="0", help="Debugger name: SEGGER for apollo (0) or ST=LINK for stm (1)")
    parser.add_argument("--dataset", default="MNIST", help="Dataset name (default: MNIST)")
    parser.add_argument("--exp-name", default="Default", help="Experiment name (default: Default)")
    parser.add_argument("--target_value", type=float, default=1.0, help="Target parameter value (default: 1.0)")
    parser.add_argument("--target_param", default="reg_alpha", help="Target parameter name (default: entropy_alpha)")
    parser.add_argument("--mlflow_port", default="8081", help="MLflow tracking port (default: 8081)")
    parser.add_argument("--mode", default=0, type=int, help="Memory mode: SORTED (0), UNSORTED (1), RANDOM SORT (2), SRAM ONLY (3)")
    parser.add_argument("--high-perf", default=0, type=int, help="High performance mode: On (0), Off (1)")
    return parser.parse_args()


def get_serial_by_id(debugger: str = "ST-LINK") -> tuple[bool, str]:
    by_id_dir = Path("/dev/serial/by-id")
    device_map = {}

    port =  ""
    if not by_id_dir.exists():
        return True, port

    for symlink in by_id_dir.iterdir():
        # resolve() follows the symlink to /dev/ttyACM*
        real_port = str(symlink.resolve())
        device_map[symlink.name] = real_port
        
        if debugger in symlink.name in symlink.name:
            port = real_port

    if port != "":
        return True, port

    return False, port

def tensor_bytes_to_c_array(tensor_bytes: bytes) -> str:
    """Deserialize int64 tensor bytes and emit C array initializer."""
    n = len(tensor_bytes) // 8
    values = struct.unpack(f"<{n}q", tensor_bytes)
    body = ", ".join(str(v) for v in values)
    return f"{{ {body} }}"


def build_header(artifacts_dir: str, header_path: str, mode: int):
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
    if mode == 2: # Random order
        li_tensor = torch.arange(len(li_tensor))[torch.randperm(len(li_tensor))]

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

def run_make(mode: int, task: str, high_perf: bool):
    print("Running make clean all...")
    perf = "HIGH_PERF=1" if high_perf else "HIGH_PERF=0" 
    if mode == 0 or mode == 2: # Optimal sorted or random sorted
        command = ["make", "clean", f"all", perf, f"TASK={task}", "SORTED=1"]
    elif mode == 1:
        command = ["make", "clean", f"all", perf, f"TASK={task}"]
    else:
        command = ["make", "clean", f"all", perf, f"TASK={task}", "SRAM=1"]
    result = subprocess.run(command, cwd="..", capture_output=True, text=True)
    if result.returncode != 0:
        print("Make failed:")
        print(result.stderr)
        return False
    print("Make successful.")
    return True

def flash_device():
    print("Flashing device with Jlink...")
    result = subprocess.run(["make", "flash"], cwd="..", capture_output=True, text=True)
    if result.returncode != 0:
        print("Flash failed:")
        print(result.stderr)
        return False
    print("Flash successful.")
    return True

def read_serial(port, timeout=90) -> str:
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
                    break
        ser.close()
    except Exception as e:
        print(f"Serial communication failed: {e}")
    return latency

def main():
    args = parse_args()
    download_dir = f"./downloaded_artifacts/{args.exp_name}/{args.dataset}_{args.target_param}_{args.target_value}"
    header_dir = f"../src/parameters/{args.dataset.lower()}_leafstats.h"
    artifact_subdir = "artifacts"
    header_filename = f"{args.exp_name}_{args.dataset}_{args.target_param}_{args.target_value}"

    # We still use MlflowClient briefly just to search for the run ID based on your criteria
    mlflow_tracking_uri = f"http://localhost:{args.mlflow_port}"
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)

    # Get experiment
    print(args.exp_name)
    experiment = client.get_experiment_by_name(args.exp_name)
    if not experiment:
        print(f"Experiment '{args.exp_name}' not found.")
        return

    # Search for the run matching the metric
    filter_string = f"params.{args.target_param} = '{args.target_value}' params.task = '{args.dataset}'"
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
    server = SERVERS[args.mlflow_port]
    for run in runs:
        seed = run.data.params["seed"]
        run_id = run.info.run_id
        artifact_uri = run.info.artifact_uri
        artifact_uri = urlparse(artifact_uri).path

        print(f"\nProcessing Run ID: {run_id} (Seed: {seed})")

        command = [
            "scp", 
            "-r", 
            f"{server}:{artifact_uri}",
            download_dir,
        ]
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"Success! Artifacts successfully downloaded to {download_dir}")
            header_path = os.path.join(download_dir, f"{header_filename}_seed{seed}.h")
            build_header(os.path.join(download_dir, artifact_subdir), header_path, args.mode)

            print(f"Copying header to {header_dir}...")
            command = [
                "cp", 
                header_path,
                f"{header_dir}",
            ]
            subprocess.run(command, capture_output=True, text=True)

            latency = "null"
            breakpoint()
            debugger = "SEGGER" if args.dbg == 0 else "ST-LINK"
            error, port = get_serial_by_id(debugger)
            if error and port != "":
                print("Directory /dev/serial/by-id does not exist.")
                assert(False)
            elif error and port == "":
                print(f"{args.dbg} not found.")
                assert(False)

            if run_make(args.mode, args.dataset, args.high_perf):
                if flash_device():
                    latency = read_serial(f"/dev/ttyACM{args.port}")
            output_file = f"results/time_{args.exp_name}_{args.dataset}_HP{args.high_perf}.csv"
            with open(output_file, "a") as f:
                f.write(f"{latency},{seed},{args.target_value},{args.mode}\n")
        else:
            print("SCP command failed with error:")
            print(result.stderr)

if __name__ == "__main__":
    main()
