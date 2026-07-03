from pathlib import Path
import torch
from torchvision.datasets import SVHN
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split

DATA_DIR = "data"

class SVHNLoader:
    def __init__(self, batch_size: int, num_workers: int, normalize):
        self.name = 'SVHN'
        self.data_dir = Path(DATA_DIR)
        
        # 1. Dataset-specific normalization values for SVHN
        # Calculated across the 3 RGB channels for the Street View House Numbers dataset
        
        # 2. Setup transforms
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize
        ])
        
        transform_eval = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        # 3. Load full raw datasets
        # Note: SVHN uses 'split' instead of 'train=True/False'
        full_train_dataset = SVHN(root=str(self.data_dir), split='train', download=True)
        test_dataset = SVHN(root=str(self.data_dir), split='test', download=True)
        
        # 4. Generate 90/10 Train and Validation split
        generator = torch.Generator().manual_seed(42) # Reproducible splits
        num_train = int(0.9 * len(full_train_dataset))
        num_val = len(full_train_dataset) - num_train
        
        train_subset, valid_subset = random_split(
            full_train_dataset, [num_train, num_val], generator=generator
        )

        # 5. Attach specific transforms dynamically using the helper class
        self.train_dataset = TransformedDataset(train_subset, transform=transform_train)
        self.valid_dataset = TransformedDataset(valid_subset, transform=transform_eval)
        self.test_dataset  = TransformedDataset(test_dataset, transform=transform_eval)

        # 6. Create Dataloaders
        self.train = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers, pin_memory=True)
        self.valid = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers, pin_memory=True)
        self.test = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, 
                               num_workers=num_workers, pin_memory=True)

        self.batch_size = batch_size
        self.in_chan = 3
        self.in_size = (32, 32)
        self.out_dim = 10  # Digits 0-9

    def get_config(self):
        return {
            "task": self.name,
            "in_chan": self.in_chan,
            "in_size": self.in_size,
            "out_dim": self.out_dim,
            "train_samples": len(self.train.dataset),
            "valid_samples": len(self.valid.dataset),
            "test_samples": len(self.test.dataset),
        }


class TransformedDataset(Dataset):
    """
    Applies transforms on the fly during training loop step execution.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.subset[index]
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.subset)
