import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
from torchaudio.datasets import SPEECHCOMMANDS 
from torch.utils.data import DataLoader
from torchaudio.datasets.utils import _extract_tar, _load_waveform
from pathlib import Path
import logging

LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "noise"]
DATA_DIR = "data/SpeechCommands"
SETS = ["train", "valid", "test"]
url = "speech_commands_v0.02"

class SCLoader:
    def __init__(self, batch_size:int, num_workers: int, feature, sample_rate:int):
        self.name = 'SC'
        self.batch_size = batch_size
        self.labels = LABELS
        self.duration = 1.0 
        self.sr = sample_rate
        self.out_dim = 10
        self.data_dir = Path(DATA_DIR)
        self.feature = feature

        # Check if data is prepared
        data_subdir = self.data_dir / url
        if not data_subdir.exists(): 
            _extract_tar(archive_file, self.data_dir)

        trainset = SubsetSC(self.data_dir.parent, "training", self.feature, self.sr)
        testset = SubsetSC(self.data_dir.parent, "testing", self.feature, self.sr)
        validset = SubsetSC(self.data_dir.parent, "validation", self.feature, self.sr)

        self.train = DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers)
        self.valid = DataLoader(validset, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers)
        self.test = DataLoader(testset, batch_size=batch_size, shuffle=False, 
                               num_workers=num_workers)

        self.batch_size = batch_size
        self.labels = LABELS
        self.out_dim = 10

        out_shape = self.feature(torch.randn(1, int(self.sr*self.duration))).shape
        self.in_chan, self.in_size = out_shape[0], tuple(out_shape[1:])
        logging.info(f"Feature size: {self.in_size}")

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
            "sample_rate": self.sr,
            "in_chan": self.in_chan,
            "in_size": self.in_size,
            "out_dim": self.out_dim,
            "feature": str(self.feature).split('(')[0],}

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, dataset_dir: Path, subset: str = "training", feature=lambda x: x,
                 sample_rate: int = 16000):
        super().__init__(dataset_dir, download=False, subset=subset)
        self.labels = LABELS
        self.sr = sample_rate
        self.duration = 1.0
        self._walker = [file for file in self._walker if file.split('/')[-2] in LABELS] 
        self.subset = subset
        self.feature = feature

    def __len__(self):
        return int(len(self._walker))

    def label_to_target(self, word):
        return torch.tensor(LABELS.index(word))

    @staticmethod
    def fix_audio_length(audio, t=1.0, sr=16000):
        '''
        audio (Tensor): Tensor of audio of dimension (1, Time)
        '''
        # Padding if needed
        req_len = int(t*sr)
        if audio.size(1) < req_len:
            p = req_len - audio.size(1)
            audio = F.pad(audio, (0, p), 'constant', 0)
        elif audio.size(1) > req_len: # Random crop
            start = np.random.randint(0, audio.size(1)-req_len)
            audio = audio[:, start:start+req_len]
        return audio

    def __getitem__(self, idx): # pyright: ignore
        waveform, _, label, _, _ = super().__getitem__(idx)
        waveform = fix_audio_length(waveform, t=self.duration, sr=self.sr)
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-10)
        feat = self.feature(waveform)
        target = self.label_to_target(label)
        return feat, target
