import logging
from pathlib import Path

from .base import BaseExp

class ExportModel(BaseExp):
    def __init__(self, out_dir: str, state_dict: str):
        super().__init__()  # Initialize BaseExp
        self.exp_name = "ExportModel"
        self.out_dir = Path(out_dir)
        self.state_dict_file = state_dict

    def get_config(self) -> dict:
        exp_conf = {'exp_name': self.exp_name,}
        return self.loader.get_config() | exp_conf 

    def run_exp(self):
        state_dict = torch.load(self.state_dict_file, map_location='cpu')
