import torch
import regex as re
import logging

from pathlib import Path

from .base import BaseTrainExp
from models.fff import FFF
from utils.fff_stats import get_leaves, get_leaf_stats


class ExportFFFStats(BaseTrainExp):
    def __init__(self, out_dir: str, state_dict: str):
        super().__init__()  # Initialize BaseTrainExp
        self.exp_name = "ExportFFFStats"
        self.stats_dir = Path(out_dir)
        self.state_dict_file = Path(state_dict)

    def get_config(self) -> dict:
        exp_conf = {'exp_name': self.exp_name,}
        return self.loader.get_config() | exp_conf 

    def get_config(self) -> dict:
        exp_conf = {'exp_name': self.exp_name,}
        return self.loader.get_config() | exp_conf 

    def get_model_config(self, state_dict_name) ->  tuple[str, int, int]:
        match = re.search(r"(\w+)_d(\d+)_l(\d+)\.pt", state_dict_name)
        if match:
            task, depth, leaf_width = match.groups()
            return task, int(depth), int(leaf_width)
        else:
            raise ValueError("Filename does not match expected pattern.")

    def run_exp(self):
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        out_filename = self.state_dict_file.stem
        state_dict = torch.load(str(self.state_dict_file), map_location='cpu', weights_only=True)
        _, depth, leaf_width = self.get_model_config(str(self.state_dict_file))
        model = FFF(in_features=self.loader.in_chan*self.loader.in_size[0]*self.loader.in_size[1], 
                    leaf_width=leaf_width, depth=depth,
                    out_features=self.loader.out_dim).to(self.device)
        model.load_state_dict(state_dict)

        val_leaves = get_leaves(model, self.loader.valid, self.device)
        val_leaf_stats = torch.tensor(get_leaf_stats(val_leaves, model.n_leaves))
        val_leaf_sorted_indices = torch.sort(val_leaf_stats, descending=True).indices

        val_new_leaf_indices = torch.empty_like(val_leaf_sorted_indices)
        val_new_leaf_indices[val_leaf_sorted_indices] = torch.arange(model.n_leaves)

        test_leaves = torch.tensor(get_leaves(model, self.loader.test, self.device)) # For testing in c

        logging.info(f"Val leaf stats: {val_leaf_stats}")
        logging.info(f"Val leaf sorted indices: {val_leaf_sorted_indices}")
        logging.info(f"Val new leaf indices: {val_new_leaf_indices}")

        torch.save(val_leaves, self.stats_dir/f"{out_filename}_val_leaves.pt") 
        torch.save(val_leaf_stats, self.stats_dir/f"{out_filename}_val_leaf_stats.pt")
        torch.save(val_leaf_sorted_indices, self.stats_dir/f"{out_filename}_val_leaves_sorted.pt")
        torch.save(val_new_leaf_indices, self.stats_dir/f"{out_filename}_val_new_leaf_indices.pt")
        torch.save(test_leaves, self.stats_dir/f"{out_filename}_test_leaves.pt") # Also test just to test

        return {}

    def log_exp(self, metrics) -> None:
        pass
