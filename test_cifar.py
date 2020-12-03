import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from torchvision import models
import torch.optim as optim

from featout.model import Net
from featout.featout_dataset import Featout

# augmentation
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

# USE NEW CLASS
trainset = Featout(
    root='./data', train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=0
)
# don't need any transformations here, so use normal testloader
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=False, num_workers=0
)

classes = (
    'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
    'truck'
)

# define model and optimizer
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(5):

    running_loss = 0.0
    blurred_set = []

    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(
                '[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 2000)
            )
            running_loss = 0.0

    # iterate over training set and select the images that were correct

    # blurr part

    # Train on extra dataset

    # Evaluate test performance
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(
        'Accuracy of the network on the 10000 test images: %d %%' %
        (100 * correct / total)
    )

# Save model
print('Finished Training')
torch.save(net.state_dict(), 'trained_models/cifar_torchvision.pt')
