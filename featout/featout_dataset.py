import torch
import torchvision
import torchvision.transforms.functional as TF
import numpy as np

from featout.interpret import simple_gradient_saliency
from featout.utils.blur import zero_out, blur_around_max


# Inherit from any pytorch dataset class
class Featout(torchvision.datasets.CIFAR10):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # initial stage: no blurring
        self.featout = False

    def __getitem__(self, index):
        image = super().__getitem__(index)
        # image = self.images[index]  # TODO: call getitem from super instead

        if self.featout:
            label = self.labels[index]  # TODO: where to get label from
            gradients = self.algorithm(self.featout_model, image, label)
            # Compute point of maximum activation
            # TODO: make the following lines more flexible, was for testing
            # TODO: add smoothing of gradients
            grads_mean = np.mean(gradients, axis=2)
            max_x = np.argmax(grads_mean.flatten()) // image.size()[1]
            max_y = np.argmax(grads_mean.flatten()) % image.size()[1]
            # blurr out and write into image variable
            image = self.blur_method(image, (max_x, max_y), patch_radius=4)
            image = TF.to_tensor(image)
        return image

    def start_featout(
        self, model, blur_method=zero_out, algorithm=simple_gradient_saliency
    ):
        """
        We can set here whether we want to blur or zero and what gradient alg
        """
        # TODO: pass predicted labels because we only do featout if it is
        # predicted correctly
        self.featout = True
        self.featout_model = model
        self.blur_method = blur_method
        self.gradient_algorithm = algorithm

    def stop_featout(self, ):
        self.featout = False


# Inspired from https://discuss.pytorch.org/t/changing-transformation-applied-to-data-during-training/15671/3
# Example usage:

# dataset = Featout(normal arguments of super dataset)
# loader = DataLoader(dataset, batch_size=2, num_workers=2, shuffle=True)
# loader.dataset.start_featout(net)
