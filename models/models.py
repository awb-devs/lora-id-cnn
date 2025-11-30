import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Multi_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)
        self.con0 = nn.Conv1d(1, 10, 3)
        self.con1 = nn.Conv2d(10, 10, (5, 2))
        self.lin0 = nn.Linear(10180, 256)
        self.lin1 = nn.Linear(256, 80)
        self.outp = nn.Linear(80, 8)

    def forward(self, x, train=False):
        # Conv Layer 0
        xI, xQ = torch.split(x, 1, dim=1)
        xI = self.con0(xI)
        xQ = self.con0(xQ)
        x = torch.stack([xI, xQ], dim=3)
        x = self.relu(x)
        # Conv Layer 1
        x = self.con1(x)
        x = self.relu(x)
        # Flatten
        x = torch.flatten(x, 1)
        # FC Layer 0
        x = self.lin0(x)
        x = self.relu(x)
        if train: x = self.drop(x)
        # FC Layer 1
        x = self.lin1(x)
        x = self.relu(x)
        if train: x = self.drop(x)
        # Output Layer:
        x = self.outp(x)
        return x # assumes CEL will handle the softmax
"""
class Multi_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)
        self.con0I = nn.Conv1d(1, 10, 3)
        self.con0Q = nn.Conv1d(1, 10, 3)
        self.con1 = nn.Conv2d(10, 10, (5, 2))
        self.lin0 = nn.Linear(10180, 256)
        self.lin1 = nn.Linear(256, 80)
        self.outp = nn.Linear(80, 8)

    def forward(self, x, train=False):
        # Conv Layer 0
        xI, xQ = torch.split(x, 1, dim=1)
        xI = self.con0I(xI)
        xQ = self.con0Q(xQ)
        x = torch.stack([xI, xQ], dim=3)
        x = self.relu(x)
        # Conv Layer 1
        x = self.con1(x)
        x = self.relu(x)
        # Flatten
        x = torch.flatten(x, 1)
        # FC Layer 0
        x = self.lin0(x)
        x = self.relu(x)
        if train: x = self.drop(x)
        # FC Layer 1
        x = self.lin1(x)
        x = self.relu(x)
        if train: x = self.drop(x)
        # Output Layer:
        x = self.outp(x)
        return x # assumes CEL will handle the softmax
    """
