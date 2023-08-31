import os
import time
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torchvision import models
import torch.optim as optim

from models.mnist_model import Net
from featout.featout_dataset import Featout
from featout.utils.blur import blur_around_max
from featout.interpret import simple_gradient_saliency

DATASET = torchvision.datasets.MNIST  # CIFAR10
# method how to remove features - here by default blurring
BLUR_METHOD = blur_around_max
# algorithm to derive the model's attention
ATTENTION_ALGORITHM = simple_gradient_saliency
# set this path to some folder, e.g., "outputs", if you want to plot the results
PLOTTING_PATH = "outputs"
if PLOTTING_PATH is not None:
    os.makedirs(PLOTTING_PATH, exist_ok=True)

# augmentation
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        # for cifar: (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

# load the original dataset (it's downloaded automatically if not found)
original_trainset = DATASET(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)
# wrap the dataset with featout
trainset = Featout(original_trainset, PLOTTING_PATH)

# the trainloader handles batching of the training set
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=0
)
# for the test data, we don't need any transformations, so we take the original
# dataset and put it into the dataloader (without shuffling)
testset = DATASET(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=False, num_workers=0
)

# define model and optimizer (standard mnist model from torch is used)
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    net.parameters(), lr=0.001, momentum=0.9
)

for epoch in range(10):
    tic = time.time()
    running_loss = 0.0
    blurred_set = []

    # START FEATOUT
    # first epoch uses unmodified dataset, then we do it every epoch
    # code could be changed to do it only every second epoch or so
    if epoch > 0:
        trainloader.dataset.start_featout(
            net,
            blur_method=BLUR_METHOD,
            algorithm=ATTENTION_ALGORITHM,
        )

    for i, data in enumerate(trainloader):
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
        if (
            i % 2000 == 1999
        ):  # print every 2000 mini-batches
            print(
                "Epoch %d, samples %5d] loss: %.3f"
                % (epoch + 1, i + 1, running_loss / 2000)
            )
            running_loss = 0.0

    print(f"time for epoch: {time.time()-tic}")

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
        "Accuracy of the network on the test images: %d %%"
        % (100 * correct / total)
    )

    # stop featout after every epoch
    trainloader.dataset.stop_featout()

# Save model
print("Finished Training")
torch.save(
    net.state_dict(), "trained_models/cifar_torchvision.pt"
)
