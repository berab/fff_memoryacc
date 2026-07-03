from pathlib import Path
import numpy as np
import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader, Dataset

DATA_DIR = "data"
CLASSES = ["dog", "bird", "truck", "ship", "airplane"]

class MappedDataset(Dataset):
    """
    A simple wrapper dataset to map original CIFAR-10 class indices (0-9)
    to a new contiguous range (0-4) and apply a specific transform.
    """
    def __init__(self, subset: Subset, class_mapping: dict, transform=None):
        self.subset = subset
        self.class_mapping = class_mapping
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.subset[index]
        # Map original CIFAR-10 label to your new CIFAR5 label (0-4)
        mapped_target = self.class_mapping[target]
        
        if self.transform:
            img = self.transform(img)
            
        return img, mapped_target

    def __len__(self):
        return len(self.subset)


class CIFAR5Loader:
    def __init__(self, batch_size: int, num_workers: int, normalize):
        self.name = 'CIFAR5'
        self.data_dir = Path(DATA_DIR)
        
        # Train-specific augmentations (applied on PIL images during loading)
        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        
        # Validation/Test transformations
        transform_eval = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        # 1. Load full raw datasets
        full_train_dataset = CIFAR10(root=str(self.data_dir), train=True, download=True)
        full_test_dataset = CIFAR10(root=str(self.data_dir), train=False, download=True)
        
        # 2. Map chosen class names to original CIFAR-10 integer targets
        cifar10_class_to_idx = full_train_dataset.class_to_idx
        target_cifar10_indices = [cifar10_class_to_idx[c] for c in CLASSES]
        
        # Mapping dictionary: e.g., {original_idx: new_idx_from_0_to_4}
        class_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(target_cifar10_indices)}

        # 3. Filter and Split Train/Valid Indices
        train_targets = np.array(full_train_dataset.targets)
        filtered_train_indices = np.where(np.isin(train_targets, target_cifar10_indices))[0]
        
        np.random.seed(42) # Reproducible validation split
        shuffled_indices = np.random.permutation(filtered_train_indices)
        num_train = int(0.9 * len(shuffled_indices))
        train_indices, val_indices = shuffled_indices[:num_train], shuffled_indices[num_train:]

        # Create basic Subsets
        train_subset = Subset(full_train_dataset, train_indices)
        valid_subset = Subset(full_train_dataset, val_indices)

        # 4. Filter Test Indices
        test_targets = np.array(full_test_dataset.targets)
        filtered_test_indices = np.where(np.isin(test_targets, target_cifar10_indices))[0]
        test_subset = Subset(full_test_dataset, filtered_test_indices)

        # 5. Wrap with MappedDataset to dynamically handle transforms and label mapping
        trains = MappedDataset(train_subset, class_mapping, transform=transform_train)
        valids = MappedDataset(valid_subset, class_mapping, transform=transform_eval)
        tests = MappedDataset(test_subset, class_mapping, transform=transform_eval)

        # 6. Create Dataloaders
        self.train = DataLoader(trains, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers, pin_memory=True)
        self.valid = DataLoader(valids, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers, pin_memory=True)
        self.test = DataLoader(tests, batch_size=batch_size, shuffle=False, 
                               num_workers=num_workers, pin_memory=True)

        self.batch_size = batch_size
        self.in_chan = 3
        self.in_size = (32, 32)
        self.out_dim = len(CLASSES) # Dynamic output dimension (5)

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
