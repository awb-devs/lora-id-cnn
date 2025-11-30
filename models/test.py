import torch
import torchvision

from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import *
from data import IQDataset

def test(name, network, dataloader):
    # Network Testing
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in tqdm(dataloader, "Testing"):
            inputs, labels = data
            outputs = network(inputs)
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += labels.size(0)
            y_true.extend(labels.numpy())
            y_pred.extend(pred.numpy())
    print('Accuracy: ', correct/total)
    return {
            'accuracy': correct/total,
            'y_true': y_true,
            'y_pred': y_pred
            }

