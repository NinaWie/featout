import numpy as np
import matplotlib.pyplot as plt
import torchvision
from captum.attr import visualization as viz
from featout.utils.utils import get_max_activation


def show_grid(images, transpose=True, save_path=None):
    """
    Plot several images with torchvision method
    """
    img = torchvision.utils.make_grid(images)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def transform_cifar(img):
    return np.transpose(
        (img.cpu().detach().numpy() / 2) + 0.5, (1, 2, 0)
    )


def get_overlayed_img(image, gradients):
    """
    Normalize gradients and overlay image with them (red channel)
    """
    normed_gradients = np.mean(gradients, axis=0)
    normed_gradients = (
        normed_gradients - np.min(normed_gradients)
    ) / (
        np.max(normed_gradients) - np.min(normed_gradients)
    )
    # Take image in greyscale
    transformed = transform_cifar(image)
    overlayed = np.tile(
        np.expand_dims(
            np.mean(transformed, axis=2).copy(), 2
        ),
        (1, 1, 3),
    )
    # colour the gradients red
    overlayed[:, :, 0] = normed_gradients
    return overlayed


def plot_together(
    image,
    gradients,
    blurred_image,
    new_grads,
    save_path="outputs/test.png",
):
    """
    Plot four images: the original one, then overlayed by gradients, then the
    blurred one, then this one overlayed by the new gradients
    """
    # get the points of max activation
    max_x, max_y = get_max_activation(gradients)
    new_max_x, new_max_y = get_max_activation(new_grads)
    # Make figure
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 4, 1)
    plt.imshow(transform_cifar(image))
    plt.title("Original input image")
    plt.subplot(1, 4, 2)
    plt.imshow(get_overlayed_img(image, gradients))
    plt.title(
        f"Model attention BEFORE blurring (max at x={max_x}, y={max_y})"
    )
    plt.subplot(1, 4, 3)
    plt.imshow(transform_cifar(blurred_image))
    plt.title("Modified input image (blurred)")
    plt.subplot(1, 4, 4)
    plt.imshow(get_overlayed_img(blurred_image, new_grads))
    plt.title(
        f"Model attention AFTER blurring (max at x={max_x}, y={max_y})"
    )
    plt.savefig(save_path)
