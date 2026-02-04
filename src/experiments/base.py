import torch
import pandas as pd
import hydra.core.hydra_config
import mlflow
import logging
from typing import Optional
from pathlib import Path
from abc import ABC, abstractmethod

from utils.mlflow import MLFlow

class BaseExp(ABC):
    def __init__(self):
        self.out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        self.log_filename = hydra.core.hydra_config.HydraConfig.get().job.name+'.log'
        self.overrides_config = self.out_dir/'.hydra/overrides.yaml'
        self.exp_name: str

    def start_run(self, seed: int):
        mlflow.log_param('seed', seed)
        mlflow.log_param("out_dir", self.out_dir)
        mlflow.log_params(self.get_config())
        mlflow.log_artifact(str(self.overrides_config))
        logging.info(f"Starting experiment {self.exp_name}...")

    def end_run(self):
        mlflow.log_param('success', True)
        mlflow.log_artifact(str(self.out_dir/self.log_filename))

        run = mlflow.active_run()
        mlflow.end_run()
        if run is not None:
            finished_run = mlflow.get_run(run.info.run_id)
            logging.info(f"MLFlow run ID: {finished_run.info.run_id}, "
                         f"status: {finished_run.info.status}")

    def end_failed_run(self, error, seed: int):
        logging.info("Failed!")
        logging.info(error)
        mlflow.log_param('success', False)

        run = mlflow.active_run()
        mlflow.end_run()
        if run is not None:
            finished_run = mlflow.get_run(run.info.run_id)
            logging.info(f"MLFlow run ID: {finished_run.info.run_id}, "
                         f"status: {finished_run.info.status}")

    def log_metrics(self, metrics) -> None:
        # Log metrics
        df = pd.DataFrame.from_dict(metrics)
        df.to_csv(self.out_dir/'metrics.csv')
        self.log_model()

    def log_epoch(self, epoch, metrics):
        for key, vals in metrics.items():
            mlflow.log_metric(key, vals[-1], step=epoch)

    def log_test(self, test_loss, test_acc):
        mlflow.log_metrics({
            'test_loss': test_loss,
            'test_acc': test_acc,
        })

    def run(self, cfg):
        self.setup(cfg.mlflow, cfg.model, cfg.loader, cfg.optim, cfg.epochs, cfg.device)
        self.start_run(cfg.seed)
        self.run_exp()
        self.end_run()

    def main(self, cfg) -> None:
        try: 
            self.run(cfg)
        except Exception as e:
            self.end_failed_run(e, cfg.seed)

    @abstractmethod
    def get_config(self) -> dict:
        pass

    def setup(self, mfwrapper: MLFlow, partial_model, loader, optim, epochs, device):
        # MLFlow setup
        mfwrapper.start()
        # Model and optim.setup
        self.model = partial_model(in_features=loader.in_chan*loader.in_size[0]*loader.in_size[1], 
                                   out_features=loader.out_dim).to(device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optim = optim(self.model.parameters())
        self.loader = loader
        self.device = device
        self.epochs = epochs

    @abstractmethod
    def run_exp(self, *args, **kwargs) -> Optional[dict]:
        pass
    
class BaseTrainExp(BaseExp):
    def __init__(self):
        super().__init__()
        self.model: torch.nn.Module

    def run(self, cfg):
        self.setup(cfg.mlflow, cfg.model, cfg.loader, cfg.optim, cfg.epochs, cfg.device)
        self.start_run(cfg.seed)
        self.log_exp(self.run_exp())
        self.end_run()

    @abstractmethod
    def log_exp(self, *args, **kwargs) -> None:
        pass

    def log_model(self) -> None:
        # Log model
        self.model.to('cpu')
        # TODO: Rename model with mlflow id then easy to follow maybe?
        torch.save(self.model, self.out_dir/'model.pt') # TODO: Add more checkpoints
        torch.save(self.model.state_dict(), self.out_dir/'state_dict.pt') # TODO: Add more checkpoints
        mlflow.log_artifact(str(self.out_dir/'model.pt'))
        mlflow.log_artifact(str(self.out_dir/'state_dict.pt'))

    def setup(self, mfwrapper: MLFlow, partial_model, loader, optim, epochs, device):
        # MLFlow setup
        mfwrapper.start()
        # Model and optim.setup
        self.model = partial_model(in_features=loader.in_chan*loader.in_size[0]*loader.in_size[1], 
                                   out_features=loader.out_dim).to(device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optim = optim(self.model.parameters())
        self.loader = loader
        self.device = device
        self.epochs = epochs
