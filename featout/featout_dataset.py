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
        image, label = super().__getitem__(index)

        if self.featout:
            # TODO: batch the gradient computing? would probably be a speed up
            in_img = torch.unsqueeze(image, 0)
            gradients = torch.squeeze(
                self.algorithm(self.featout_model, in_img, label)
            ).numpy()
            # Compute point of maximum activation
            # TODO: make the following lines more flexible, was for testing
            # TODO: add smoothing of gradients
            grads_mean = np.mean(gradients, axis=0)
            max_x = np.argmax(grads_mean.flatten()) // grads_mean.shape[1]
            max_y = np.argmax(grads_mean.flatten()) % grads_mean.shape[1]
            # blurr out and write into image variable
            image = torch.squeeze(
                self.blur_method(in_img, (max_x, max_y), patch_radius=4)
            )
            # TODO: test by saving the image before and after
        return image, label

    def start_featout(
        self,
        model,
        blur_method=blur_around_max,
        algorithm=simple_gradient_saliency
    ):
        """
        We can set here whether we want to blur or zero and what gradient alg
        """
        # TODO: pass predicted labels because we only do featout if it is
        # predicted correctly
        print("start featout")
        self.featout = True
        self.algorithm = simple_gradient_saliency
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
