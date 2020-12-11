from featout.utils.gaussian_smoothing import GaussianSmoothing
import torch.nn.functional as F


def blur_around_max(img, max_coordinates, patch_radius=4, kernel_size=5):
    """
    Blur a quadratic patch around max_coordinates with a Gaussian filter
    """
    max_x, max_y = max_coordinates
    modified_input = img.clone()

    # check whether it is in the image bounds
    x_start = max([max_x - patch_radius, 0])
    x_end = min([max_x + patch_radius + 1, modified_input.shape[2]])
    y_start = max([max_y - patch_radius, 0])
    y_end = min([max_y + patch_radius + 1, modified_input.shape[3]])

    patch = modified_input[:, :, x_start:x_end, y_start:y_end]

    # smooth only the patch (padded)
    # TODO: instead of padding use real surroundings of patch
    smoothing = GaussianSmoothing(3, kernel_size, 1)
    patch_pad = F.pad(patch, (2, 2, 2, 2), mode='reflect')
    smoothed_patch = smoothing(patch_pad)

    modified_input[:, :, x_start:x_end, y_start:y_end] = smoothed_patch
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
