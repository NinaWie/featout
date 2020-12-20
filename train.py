import time
import numpy as np
import json
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import torch.optim as optim

from models.mnist_model import Net as mnist_net
from models.cifar_model import Net as cifar_net
from featout.featout_dataset import Featout
from featout.interpret import simple_gradient_saliency
from featout.utils.blur import zero_out, blur_around_max

DATASET_INFOS = {
    "mnist":
        {
            "load": torchvision.datasets.MNIST,
            "normalize": ((0.1307, ), (0.3081, )),
            "model": mnist_net
        },
    "cifar":
        {
            "load": torchvision.datasets.CIFAR10,
            "normalize": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            "model": cifar_net
        }
}
# decide which dataset to use
DATASET = "cifar"

NR_EPOCHS = 20
NR_RUNS = 10
RADIUS = 3
INTERPRET = simple_gradient_saliency
BLUR_OR_ZERO = blur_around_max

MODEL_PATH = "trained_models/cifar"
ID = "m"
DO_FEATOUT = 0

data_info = DATASET_INFOS[DATASET]

# augmentation
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(*data_info["normalize"])]
)

# take cifar
original_trainset = data_info["load"](
    root='./data', train=True, download=True, transform=transform
)
# and wrap with featout
trainset = Featout(original_trainset)
# TODO: shuffle set to false for tests --> change back
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=False, num_workers=0
)
# don't need any transformations here, so use normal testloader
testset = data_info["load"](
    root='./data', train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=False, num_workers=0
)

for j in range(NR_RUNS):
    # define model and optimizer
    net = data_info["model"]()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # run for x epochs
    losses, train_accs, test_accs, runtimes = list(), list(), list(), list()
    for epoch in range(NR_EPOCHS):
        tic = time.time()
        running_loss = 0.0
        blurred_set = []

        # iterate over training set and select the images that were correct
        if DO_FEATOUT and epoch > 0:  # and epoch % 2 == 0:
            trainloader.dataset.start_featout(
                net,
                blur_method=BLUR_OR_ZERO,
                algorithm=INTERPRET,
                patch_radius=RADIUS
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
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    '[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000)
                )
                losses.append(running_loss / 2000)
                running_loss = 0.0

        print(f"time for epoch: {time.time()-tic}")
        runtimes.append(time.time() - tic)

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
        test_accs.append(100 * correct / total)

        # stop featout
        trainloader.dataset.stop_featout()

    # Save model
    print('Finished Training')
    torch.save(
        net.state_dict(),
        os.path.join(MODEL_PATH, f'{ID}_feat_{DO_FEATOUT}_num_{j}.pt')
    )
    # save stats
    res_dict = {
        "losses": losses,
        "test_acc": test_accs,
        "train_acc": train_accs,
        "runtimes": runtimes
    }
    with open(
        os.path.join(MODEL_PATH, f'{ID}_feat_{DO_FEATOUT}_num_{j}.json'), "w"
    ) as outfile:
        json.dump(res_dict, outfile)
    print("Saved successfully with name", f'{ID}_feat_{DO_FEATOUT}_num_{j}')
    print()
