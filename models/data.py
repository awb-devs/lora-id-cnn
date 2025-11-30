import torch
import numpy as np
from typing import List, Tuple
from numpy.typing import NDArray as Mat
from numpy import float32 as f32
from numpy import complex64 as c64
from pathlib import Path
from torch.utils.data import Dataset

class IQDataset(Dataset):
    def __init__(self, 
                 dataset,
                 datatype,
                 root='dataset',
                 devices=['esp32A', 'esp32B', 'esp32C', 'esp32D', 'esp32E', 'esp32F'],
                 window=False,
                 window_len=0,
                 transform=None, 
                 target_transform=None):
        self.labels = labels_from_dir(root, devices, dataset, datatype)
        self.transform = transform
        self.target_transform = target_transform
        self.classes = devices

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label, path = self.labels[idx]
        buf = np.fromfile(path, dtype=c64)
        buf = np.stack([buf.real, buf.imag], axis=0)
        if self.transform: 
            buf = self.transform(buf)
        if self.target_transform: 
            buf = self.target_transform(label)
        return buf, label

def labels_from_dir(path: str, devices: List[str], data_class: str, train: str) -> List[Tuple[int, Path]]:
    labels = []
    for label, device in enumerate(devices):
        subpath = Path(f"{path}/{data_class}/{device}/{train}")
        print(subpath)
        for i, file in enumerate(subpath.glob('*.cfile')):
            labels.append((label, file))
    return labels

