import os
import mlflow
from mlflow.tracking import MlflowClient
import subprocess
from urllib.parse import urlparse
import struct
import serial
import time

# 1. Configure the MLflow tracking URI and experiment details
MLFLOW_TRACKING_URI = "http://localhost:8083"
DEVICE = "STM"
DATASET = "mnist"
EXPERIMENT_NAME = f"{DEVICE}_{DATASET}_iafff_hard_seed"
TARGET_PARAM = "entropy_alpha"
TARGET_VALUE = 1.0
DOWNLOAD_DIR = f"./downloaded_artifacts/{EXPERIMENT_NAME}/{DEVICE}_{DATASET}_{TARGET_PARAM}_{TARGET_VALUE}"
HEADER_DIR = f"../Core/Inc/parameters/{DATASET}_leafstats.h"
ARTIFACT_SUBDIR = "artifacts"
HEADER_FILENAME = f"{DEVICE}_{DATASET}_{TARGET_PARAM}_{TARGET_VALUE}"
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 9600

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

    guard = f"{DEVICE}_{DATASET}_{TARGET_PARAM}_{TARGET_VALUE}_H".upper().replace(".", "_").replace("-", "_")

    header = f"""#define LT {lt_arr}
#define LI {li_arr}
"""
    with open(header_path, "w") as f:
        f.write(header)
    print(f"Header written to {header_path}")
    return True

def run_make():
    print("Running make clean all...")
    result = subprocess.run(["make", "clean", "all"], cwd="..", capture_output=True, text=True)
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

def read_serial(output_file, timeout=15):
    print(f"Reading serial output from {SERIAL_PORT}...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        with open(output_file, "w") as f:
            start_time = time.time()
            while time.time() - start_time < timeout:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    print(f"Serial: {line}")
                    f.write(line + "\n")
                    f.flush()
                    if "elapsed_ms=" in line:
                        break
        ser.close()
        print(f"Serial output saved to {output_file}")
        return True
    except Exception as e:
        print(f"Serial communication failed: {e}")
        return False

def main():
    # We still use MlflowClient briefly just to search for the run ID based on your criteria
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # Get experiment
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        print(f"Experiment '{EXPERIMENT_NAME}' not found.")
        return

    # Search for the run matching the metric
    filter_string = f"params.{TARGET_PARAM} = '{TARGET_VALUE}'"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        max_results=5
    )

    if not runs:
        print(f"No runs found matching {TARGET_PARAM} = {TARGET_VALUE}")
        return

    print(f"Found {len(runs)} matching run(s).")
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
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
        DOWNLOAD_DIR,
        ]
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"Success! Artifacts successfully downloaded to {DOWNLOAD_DIR}")
            header_path = os.path.join(DOWNLOAD_DIR, f"{HEADER_FILENAME}_seed{seed}.h")
            build_header(os.path.join(DOWNLOAD_DIR, ARTIFACT_SUBDIR), header_path)
            
            print(f"Copying header to {HEADER_DIR}...")
            command = [
            "cp", 
            header_path,
            f"{HEADER_DIR}",
            ]
            subprocess.run(command, capture_output=True, text=True)

            if run_make():
                if flash_device():
                    serial_output_file = f"results/serial_output_seed{seed}.txt"
                    read_serial(serial_output_file)
        else:
            print("SCP command failed with error:")
            print(result.stderr)

if __name__ == "__main__":
    main()
