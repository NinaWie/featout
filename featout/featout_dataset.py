import torch
import numpy as np
import os

from featout.utils.utils import get_max_activation
from featout.interpret import simple_gradient_saliency
from featout.utils.blur import zero_out, blur_around_max
from captum.attr import visualization as viz
from featout.utils.plotting import plot_together


class Featout(torch.utils.data.Dataset):
    """
    Inspired from https://discuss.pytorch.org/t/changing-transformation-applied-to-data-during-training/15671/3
    Example usage:
    dataset = Featout(normal arguments of super dataset)
    loader = DataLoader(dataset, batch_size=2, num_workers=2, shuffle=True)
    loader.dataset.start_featout(net)
    """

    def __init__(
        self, dataset, plotting_path, *args, **kwargs
    ):
        """
        Args:
            dataset: torch Dataset object (must impelemnt getitem and len)
        """
        # actual dataset
        self.dataset = dataset
        # initial stage: no blurring
        self.featout = False
        # set path where to save plots (set to None if no plotting desired)
        self.plotting = plotting_path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Main workflow: Get image, if label correct, then featout (blur/zero)
        and return the modified image
        """
        # call method from base dataset
        image, label = self.dataset.__getitem__(index)

        if self.featout:
            in_img = torch.unsqueeze(image, 0)

            # run a prediction with the given model --> TODO: this can be done
            # more efficiently by passing the predicted labels from the
            # preceding epoch to this class
            _, predicted_lab = torch.max(
                self.featout_model(in_img).data, 1
            )
            # only do featout if it was predicted correctly
            if predicted_lab == label:
                # get model attention via gradient based method
                gradients = self.algorithm(
                    self.featout_model, in_img, label
                )[0].numpy()
                # Compute point of maximum activation
                max_x, max_y = get_max_activation(gradients)

                # blur patch at activation (feat-out)
                blurred_image = self.blur_method(
                    in_img, (max_x, max_y), patch_radius=4
                )
                # save images before and after if plotting is desired
                if self.plotting is not None:
                    new_grads = self.algorithm(
                        self.featout_model,
                        blurred_image,
                        label,
                    )[0].numpy()
                    plot_together(
                        image,
                        gradients,
                        blurred_image[0],
                        new_grads,
                        save_path=os.path.join(
                            self.plotting,
                            f"images_{index}.png",
                        ),
                    )

                image = blurred_image[0]

        return image, label

    def start_featout(
        self,
        model,
        blur_method=blur_around_max,
        algorithm=simple_gradient_saliency,
    ):
        """
        We can set here whether we want to blur or zero and what gradient alg
        """
        print("\n STARTING FEATOUT \n ")
        self.featout = True
        self.algorithm = algorithm
        self.featout_model = model
        self.blur_method = blur_method

    def stop_featout(self):
        self.featout = False
