import torch
from pathlib import Path

from .base import BaseExp

class ExportIntermediates(BaseExp):
    def __init__(self, out_dir: str, model):
        super().__init__()  # Initialize BaseExp
        self.exp_name = "ExportIntermediates"
        self.int_dir = Path(out_dir)
        self.model_name = model

    def get_config(self) -> dict:
        exp_conf = {'exp_name': self.exp_name,}
        return self.loader.get_config() | exp_conf 

    def run_exp(self):
        model = torch.load(self.model_name)
        task = self.loader.name.lower()

        out_dir = self.int_dir/task
        out_dir.mkdir(parents=True, exist_ok=True)

        samples, targets = next(iter(self.loader.test))
        model.save_intermediates(samples, out_dir)
        torch.save(samples[0], out_dir/"input.pt")
        torch.save(targets[0], out_dir/"target.pt")
