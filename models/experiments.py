import torch
import torchvision

from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path

from train import train
from test import test
from data import IQDataset
from report import generate_report
import models

"""
Experiment Index

1.  Basic experiment, testing the model parameters as described in the paper, 
    but with the full 1024 preamble buffers.

2.  Outdoor experiment, with the full 1024 preamble buffers.

3.  Effect of decimation on classification accuricy

4.  Effect of windowing of various sizes on classification accuracy.

5.  Effect of signal to noise ratio from artificial noise.

6.  Binary classification with untrained input

7.  Live demo model

"""

"""
Experiment 1
Basic Model Trained on Wired Data
"""
"""
name = "Exp01"
dataset = 'wired_250_24'
batch_size = 200
lr = 0.001
epochs = 50

tr_data = IQDataset(dataset, 'train')
tr_dataloader = DataLoader(tr_data, batch_size, shuffle=True)

ts_data = IQDataset(dataset, 'test')
ts_dataloader = DataLoader(ts_data, batch_size, shuffle=True)

network = models.Multi_Net()
optim = optim.Adam(network.parameters(), lr)

Path(name).mkdir(exist_ok=True)

loss = train(name, network, optim, tr_dataloader, epochs)
results = test(name, network, ts_dataloader)
generate_report(
        name,
        dataset,
        lr,
        epochs,
        "",
        loss,
        results,
        ts_data.classes
        )
"""
"""
Experiment 2
Basic Model Trained on Room Data
"""

name = "Exp02-room"
dataset = 'room_250_16'
batch_size = 200
lr = 0.001
epochs = 50

tr_data = IQDataset(dataset, 'train')
tr_dataloader = DataLoader(tr_data, batch_size, shuffle=True)

ts_data = IQDataset(dataset, 'test')
ts_dataloader = DataLoader(ts_data, batch_size, shuffle=True)

network = models.Multi_Net()
optim = optim.Adam(network.parameters(), lr)

Path(name).mkdir(exist_ok=True)

loss = train(name, network, optim, tr_dataloader, epochs)
results = test(name, network, ts_dataloader)
generate_report(
        name,
        dataset,
        lr,
        epochs,
        "",
        loss,
        results,
        ts_data.classes
        )

