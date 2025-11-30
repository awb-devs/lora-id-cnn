import torch
import numpy as np
from typing import List, Tuple
from numpy.typing import NDArray as Mat
from numpy import float32 as f32
from numpy import complex64 as c64
from pathlib import Path
from torch.utils.data import Dataset

"""
todo:
    - docs
    - move random sampling to python script
    - windowing
    - decimation
"""

def generate_labelsets(
        dataset: str,
        root='dataset',
        devices=['esp32A', 'esp32B', 'esp32C', 'esp32D', 'esp32E', 'esp32F'],
        trainN=600,
        testN=400,
        ):
    """Create label sets by random sampling"""
    train_labels: List[Tuple[int, Path]] = []
    test_labels: List[Tuple[int, Path]] = []
    for label, device in enumerate(devices):
        device_labels = []
        subpath = Path(f"{root}/{dataset}/{device}")
        for i, file in enumerate(subpath.glob('*.cfile')):
            device_labels.append((label, file))
        if len(device_labels) < trainN+testN:
            print(f"Warning: insufficient samples for {device}. Expected {testN + trainN}, got {len(device_labels)}. Test and or train sets may be underpopulated.")
        nplabels = np.array(device_labels)
        np.random.shuffle(nplabels)
        device_tr_labels = nplabels[:trainN]
        device_ts_labels = nplabels[trainN:trainN+testN]
        train_labels.extend(device_tr_labels)
        test_labels.extend(device_ts_labels)
    return np.array(train_labels), np.array(test_labels)

class IQDataset(Dataset):
    def __init__(self,
                 labels,
                 classes=['txA', 'txB', 'txC', 'txD', 'txE', 'txF'],
                 noise=0.0,
                 window=None,
                 decimation=1,
                 transform=None,
                 target_transform=None
                 ):
        self.labels = labels
        self.classes = classes
        self.transform = transform
        self.target_transform = target_transform
        self.noise = noise
        self.window = window
        self.decimation = decimation

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # get raw samples
        label, path = self.labels[idx]
        buf = np.fromfile(path, dtype=c64)
        buf = np.stack([buf.real, buf.imag], axis=0)
        if self.noise != 0.0:
            pass
        if self.window is not None:
            pass
        if self.decimation != 1:
            buf = buf[::self.decimation]
        if self.transform:
            buf = self.transform(buf)
        if self.target_transform:
            buf = self.target_transform(label)
        return buf, label

"""
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
"""
