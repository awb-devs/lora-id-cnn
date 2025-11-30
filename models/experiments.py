import numpy as np
import torch
import torchvision
import sys
import argparse

from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path


from train import train
from test import test
from data import generate_labelsets, IQDataset
from report import generate_report
import models


experiments = {
        'create_labelsets': False,
        'wired_basic': False,
        'room_basic': False
}

args = sys.argv[1:]
for arg in args:
    experiments[arg] = True
    
if experiments['create_labelsets']:
    wired_tr, wired_ts = generate_labelsets('wired_250_24')
    room_tr, room_ts = generate_labelsets('room_250_16')
    np.savez('dataset/labels.npz', wired_tr=wired_tr, wired_ts=wired_ts, room_tr=room_tr, room_ts=room_ts)

if experiments['wired_basic']:
    name = 'wired_basic'
    dataset = 'wired_250_24'
    batch_size = 200
    lr = 0.001
    epochs = 50

    stored_labels = np.load('dataset/labels.npz', allow_pickle=True)
    tr_labels = stored_labels['wired_tr']
    ts_labels = stored_labels['wired_ts']
    
    tr_data = IQDataset(tr_labels)
    ts_data = IQDataset(ts_labels)

    tr_loader = DataLoader(tr_data, batch_size, shuffle=True)
    ts_loader = DataLoader(ts_data, batch_size, shuffle=True)
    
    network = models.Multi_Net()
    optim = optim.Adam(network.parameters(), lr)

    Path(name).mkdir(exist_ok=True)

    loss = train(name, network, optim, tr_loader, epochs)
    results = test(name, network, ts_loader)
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

if experiments['room_basic']:
    name = 'room_basic'
    dataset = 'room_250_16'
    batch_size = 200
    lr = 0.001
    epochs = 50

    stored_labels = np.load('dataset/labels.npz', allow_pickle=True)
    tr_labels = stored_labels['room_tr']
    ts_labels = stored_labels['room_ts']
    
    tr_data = IQDataset(tr_labels)
    ts_data = IQDataset(ts_labels)

    tr_loader = DataLoader(tr_data, batch_size, shuffle=True)
    ts_loader = DataLoader(ts_data, batch_size, shuffle=True)

    network = models.Multi_Net()
    optim = optim.Adam(network.parameters(), lr)

    Path(name).mkdir(exist_ok=True)

    loss = train(name, network, optim, tr_loader, epochs)
    results = test(name, network, ts_loader)
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


