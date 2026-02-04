import torch
import random
import hydra
# import mlflow
from omegaconf import DictConfig
from dataclasses import dataclass
from hydra.utils import instantiate
from hydra.experimental.callback import Callback
from typing import Any

# class MLFlowToCSV(Callback):
#     def __init__(self, exp):
#         self.exp = exp
#
#     def on_multirun_end(self, config: DictConfig, **kwargs: Any):
#         mlflow.set_tracking_uri("file:data/mlruns")
#         experiment = mlflow.get_experiment_by_name(self.exp)
#
#         if experiment is not None:
#             runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
#             os.makedirs("data/csv_files", exist_ok=True)
#             runs_df.to_csv(f"data/csv_files/{self.exp}.csv", index=False)

@dataclass
class ENV:
    vars: list[str]
    vals: list[str]

@dataclass
class MLFlowType:
    token: str
    exp: str
    username: str
    env: ENV

@dataclass
class Main:
    exp: object
    model: torch.nn.Module
    loader: object
    optim: torch.optim.Optimizer
    mlflow: MLFlowType
    device: torch.device
    proj_name: str
    epochs: int
    seed: int

@hydra.main(config_path="../conf/", config_name="main", version_base='1.3')
def main(cfg: DictConfig):
    # Init RNG_level s
    random.seed(cfg.seed)
    cfg = instantiate(cfg)
    # Run experiment
    # cfg.exp.main(cfg)
    cfg.exp.run(cfg)

if __name__ == "__main__":
    main()
