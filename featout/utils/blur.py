from utils.gaussian_smoothing import GaussianSmoothing
import torch.nn.functional as F


def blur_around_max(img, max_coordinates, patch_radius=4, kernel_size=5):
    """
    Blur a quadratic patch around max_coordinates with a Gaussian filter
    """
    max_x, max_y = max_coordinates
    modified_input = img.clone()
    patch = modified_input[:, :, max_x - patch_radius:max_x + patch_radius + 1,
                           max_y - patch_radius:max_y + patch_radius + 1]

    # smooth only the patch (padded)
    # TODO: instead of padding use real surroundings of patch
    smoothing = GaussianSmoothing(3, kernel_size, 1)
    patch_pad = F.pad(patch, (2, 2, 2, 2), mode='reflect')
    smoothed_patch = smoothing(patch_pad)

    modified_input[:, :, max_x - patch_radius:max_x + patch_radius + 1, max_y -
                   patch_radius:max_y + patch_radius + 1] = smoothed_patch
    return modified_input


def zero_out(img, max_coordinates, patch_radius=4):
    """
    Zero out a quadratic patch around max_coordinates with a Gaussian filter
    """
    max_x, max_y = max_coordinates
    modified_input = img.clone()
    modified_input[:, :, max_x - patch_radius:max_x + patch_radius + 1,
                   max_y - patch_radius:max_y + patch_radius + 1] = 0
    return modified_input
