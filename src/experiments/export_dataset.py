import torch
from pathlib import Path

from .base import BaseExp

class ExportDataset(BaseExp):
    def __init__(self, out_dir: str):
        super().__init__()  # Initialize BaseExp
        self.exp_name = "ExportDataset"
        self.bin_dir = Path(out_dir)
        self.sample_dir = Path("data/samples")

    def get_config(self) -> dict:
        exp_conf = {'exp_name': self.exp_name,}
        return self.loader.get_config() | exp_conf 

    def run_exp(self):
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        set_name = self.loader.name.lower()

        in_features = (self.loader.in_chan * self.loader.in_size[0] 
                       * self.loader.in_size[1])
        with open(self.bin_dir/f"{set_name}.h", 'w') as f:
            f.write(f"#define IN_FEATURES {in_features}\n")
            f.write(f"#define OUT_FEATURES {self.loader.out_dim}\n")
            f.write(f"#define N_SAMPLES {self.loader.batch_size}\n")
            f.write(f"#define FILENAME {set_name}.bin\n")

        # single sample
        input, _ = next(iter(self.loader.test))
        input = input[0].flatten()
        torch.save(input, self.sample_dir/f"{set_name}.pt")
        with open(self.sample_dir/f"{set_name}_sample.h", 'wb') as f:
            input = str(input.tolist()).replace("[", "{").replace("]", "}")
            f.write(f"#define INPUT {input}\n".encode())

        for _, (input, _) in enumerate(self.loader.test):
            with open(self.bin_dir/f"{set_name}.bin", 'wb') as f:
                f.write(input.flatten(1).numpy().tobytes())

