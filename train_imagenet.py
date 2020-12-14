import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torch.utils.data as data

import torchvision.transforms as transforms

from torch.autograd import Variable

import math
import os

import logging

LEVEL = logging.DEBUG

logger = logging.getLogger(__name__)

logger.setLevel(LEVEL)
ch = logging.StreamHandler()
ch.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(ch)

from models.imagenet_model import ResNet, Bottleneck, BasicBlock

MODEL_NAME = "test"
BATCH_SIZE = 32

if __name__ == "__main__":
    net = ResNet(Bottleneck, [3, 4, 6, 3])

    # loss function + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    # load data set
    logger.info("Reading data...")
    train_dir = 'data/tiny-imagenet-200/train'
    train_dataset = datasets.ImageFolder(
        train_dir, transform=transforms.ToTensor()
    )
    train_loader = data.DataLoader(
        train_dataset, shuffle=True, batch_size=BATCH_SIZE
    )
    logger.info("Loaded: %s", train_dir)

    # for i, data in enumerate(train_loader, 0):
    #     input, target = data
    #     print(input.size())
    #     print(Variable(target))
    #     if i > 10:
    #         print(fail)
    NUM_CLASSES = len(train_dataset.classes)
    print(NUM_CLASSES)
    # train the model
    for epoch in range(2):
        logger.info("-- EPOCH: %s", epoch)
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            if i % 50 == 49:
                logger.info("-- ITERATION: %s", i)
            input, target = data

            # wrap input + target into variables
            input_var = Variable(input)
            target_var = Variable(target)

            # compute output
            output = net(input_var)
            # targets_one_hot = torch.nn.functional.one_hot(
            #     target, num_classes=NUM_CLASSES
            # )
            loss = criterion(output, target)

            # computer gradient + sgd step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print progress
            running_loss += loss.item()

            if i % 50 == 49:  # print every 2k mini-batches
                logger.info("-- RUNNING_LOSS: %s", running_loss / 50)
                running_loss = 0.0

    logger.info('Finished Training')
    torch.save(net, "trained_models/imagenet/" + MODEL_NAME)
