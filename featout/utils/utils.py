import numpy as np
from scipy.signal import convolve2d


def get_max_activation(gradients, filter_size=3):
    """
    Get the coordinates where the activation is maximal
    Includes smoothing with an all-ones filter of size filter_size
    """
    grads_mean = np.mean(gradients, axis=0)
    # smooth the results to avoid using outlier activation
    filtered = convolve2d(
        grads_mean,
        np.ones((filter_size, filter_size)),
        mode="same",
    )
    # get max of smoothed array
    max_act = np.argmax(filtered.flatten())
    # get corresponding x and y coordinates
    max_x = max_act // grads_mean.shape[1]
    max_y = max_act % grads_mean.shape[1]
    return max_x, max_y
