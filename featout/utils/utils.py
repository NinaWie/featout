import numpy as np


def get_max_activation(gradients):
    """
    Get the coordinates where the activation is maximal
    """
    # TODO: make the following lines more flexible, was for testing
    # TODO: add smoothing of gradients
    grads_mean = np.mean(gradients, axis=0)
    max_x = np.argmax(grads_mean.flatten()) // grads_mean.shape[1]
    max_y = np.argmax(grads_mean.flatten()) % grads_mean.shape[1]
    return max_x, max_y