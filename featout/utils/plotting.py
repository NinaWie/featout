import numpy as np
import matplotlib.pyplot as plt
import torchvision
from captum.attr import visualization as viz


def show_grid(images, transpose=True, save_path=None):
    img = torchvision.utils.make_grid(images)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def transform_cifar(img):
    return np.transpose((img.cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))
