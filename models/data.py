import torch
import numpy as np
import random
from typing import List, Tuple
from numpy.typing import NDArray as Mat
from numpy import float32 as f32
from numpy import complex64 as c64
from pathlib import Path
from torch.utils.data import Dataset

"""
Creates test and train labelsets which contain lists of (label, path_to_sample) pairs. 
Paths to samples are expected be in <root>/<dataset>/<device>.
Note: if trainN + testN is greater than the number of samples, the function will issue a warning,
    but still attempt to fill first the training set then the testingset. 
Parameters:
    dataset: a string which is the name of the dataset.
    root: the root path. 
    devices: the device name.
    trainN: the number of samples in the training set.
    testN: the number of samples in the testing set.
Returns:
    numpy arrays for the training and testing label sets.
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

"""
Class which implements a pytorch dataset subclass to interface with the IQ Sample Data.

Constructor:
    labels: the label set to use as returned by generate_labelsets.
        should be tuples with (label, path_to_sample)
    classes: the classes to use for chart labeling, etc. 
    noise: the desired signal to noise ratio in dB.
    window: the window length desired in frames.
    window_offset: the offset from 0 to start the window.
        You could also give 'rand' here to randomly select the offset.
    decimation: the amount to decimate the signal by.
        this means only selecting every d samples where d is the decimation.
    normalized: whether or not to normalize the signal.
    transform: transform if applicabel for the signal. 
    target_transform: transform if applicable for the labels.

Note:
    the order of effects applied is:
    decimation -> windowing -> normalization -> noise -> transforms

Len: return the number of available samples.
GetItem: get a single sample at the given index.
    Effects are applied at this stage
"""
class IQDataset(Dataset):
    def __init__(self,
                 labels,
                 classes=['txA', 'txB', 'txC', 'txD', 'txE', 'txF'],
                 noise=None,
                 window=None,
                 window_offset=0,
                 decimation=1,
                 normalized=False,
                 transform=None,
                 target_transform=None
                 ):
        self.labels = labels
        self.classes = classes
        self.noise = noise
        self.window = window
        self.window_offset = window_offset
        self.decimation = decimation
        self.normalized = normalized
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # get raw samples
        label, path = self.labels[idx]
        buf = np.fromfile(path, dtype=c64)
        buf = np.stack([buf.real, buf.imag], axis=0)
        if self.decimation != 1:
            buf = buf[:, ::self.decimation]
        if self.window is not None and isinstance(self.window_offset, int):
            buf = buf[:, self.window_offset:self.window_offset+self.window]
        if self.window is not None and self.window_offset == 'rand':
            offset = random.randrange(buf.shape[1] - self.window)
            buf = buf[:, offset:offset+self.window]
        if self.normalized:
            buf = buf / np.linalg.norm(buf) 
        if self.noise is not None:
            signal_power = np.mean(buf**2)
            snr_linear = 10**(self.noise / 10)
            noise_power = signal_power / snr_linear
            noise_amp = np.sqrt(noise_power)
            noise = np.random.normal(0, noise_amp, buf.shape).astype(f32)
            buf = buf + noise        
        if self.transform:
            buf = self.transform(buf)
        if self.target_transform:
            label = self.target_transform(label)
        return buf, label

