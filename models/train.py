import torch
import torchvision

from tqdm import tqdm
import torch.nn as nn

def train(name, network, optimizer, dataloader, epochs):
    criterion = nn.CrossEntropyLoss()
    training_loss = []
    # Training
    for epoch in tqdm(range(epochs), f"Training..."):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = network(inputs, train=True)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        training_loss.append(running_loss)

    torch.save(network.state_dict(), f"{name}/{name}.pth")
    return training_loss
