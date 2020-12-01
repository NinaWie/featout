from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
import numpy as np


def simple_gradient_saliency(net, input_img, label):
    """
    Simplest interpretable method
    Takes an image and computes the gradients
    """
    initial_input = input_img.clone()
    # activate gradients
    initial_input.requires_grad = True
    net.eval()
    # saliency method --> can use other method here
    saliency = Saliency(net)
    grads = saliency.attribute(initial_input, target=label)
    return grads


# OTHER INTERPRETABLE METHODS


def attribute_image_features(net, input_img, label, algorithm, **kwargs):
    """
    Run different interpretable algorithms in one method
    """
    initial_input = input_img.clone()
    # activate gradients
    initial_input.requires_grad = True

    net.zero_grad()
    tensor_attributions = algorithm.attribute(
        input_img, target=label, **kwargs
    )

    return tensor_attributions


def integrated_gradients(net, input_img, label):
    ig = IntegratedGradients(net)
    attr_ig, delta = attribute_image_features(
        net,
        input_img,
        label,
        ig,  # this is the algorithm
        baselines=input * 0,
        return_convergence_delta=True
    )
    print('Approximation delta: ', abs(delta))
    return attr_ig
