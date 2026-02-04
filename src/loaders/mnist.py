from pathlib import Path
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader, TensorDataset

DATA_DIR = "data/MNIST"
SETS = ["train", "valid", "test"]

class MNISTLoader:
    def __init__(self, batch_size: int, num_workers: int, normalize):
        self.name = 'MNIST'
        self.data_dir = Path(DATA_DIR)
        transform = transforms.Compose([transforms.ToTensor(), normalize])

        whole_tset = MNIST(root=self.data_dir.parent, train=True, download=True,
                                 transform=transform)
        testset = MNIST(root=self.data_dir.parent, train=False, download=True, 
                        transform=transform)

        perm = torch.randperm(len(whole_tset))
        val_len = int(len(perm)*0.1) # 10% for validation
        trainset = Subset(whole_tset, perm[val_len:].tolist())
        validset = Subset(whole_tset, perm[:val_len].tolist())

        self.train = DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers)
        self.valid = DataLoader(validset, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers)
        self.test = DataLoader(testset, batch_size=batch_size, shuffle=False, 
                               num_workers=num_workers)

        self.batch_size = batch_size

        self.in_chan = 1
        self.in_size = (28, 28)
        self.out_dim = 10

    def get_tensors(self, sets):
        for i, dset in enumerate(SETS):
            images, labels = [], []
            for img, label in sets[i]:
                images.append(img)
                labels.append(torch.tensor(label))
            torch.save(torch.stack(images), self.data_dir/f"{dset}_tensors.pt") 
            torch.save(torch.stack(labels), self.data_dir/f"{dset}_labels.pt")

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
