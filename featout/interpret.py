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
