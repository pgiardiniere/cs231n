from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        ############################################################################
        F, C, H, W = num_filters, *input_dim  # noqa
        # FH, FW: Filter height/width. PH, PW: Pool height/width.
        FH, FW = filter_size, filter_size
        PH: int = 1 + (H - 2) // 2
        PW: int = 1 + (W - 2) // 2

        # w1 is our convolutional layer, w2 & w3 are affine layers (b's similarly).
        #
        # Per def given in docstring, there's a pooling layer after the conv layer,
        # so adjust size of w2 accordingly.
        w1 = weight_scale * np.random.randn(F, C, FH, FW)
        b1 = np.zeros(F)  # Breaks the pattern, but fast_forward expects it this way.
        w2 = weight_scale * np.random.randn(F * PH * PW, hidden_dim)
        b2 = np.zeros(hidden_dim)
        w3 = weight_scale * np.random.randn(hidden_dim, num_classes)
        b3 = np.zeros(num_classes)

        self.params["W1"], self.params["W2"], self.params["W3"] = w1, w2, w3
        self.params["b1"], self.params["b2"], self.params["b3"] = b1, b2, b3

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, x, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        w1, b1 = self.params["W1"], self.params["b1"]
        w2, b2 = self.params["W2"], self.params["b2"]
        w3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = w1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        scores, cv_cache = conv_relu_pool_forward(x, w1, b1, conv_param, pool_param)
        scores, fc_1_cache = affine_relu_forward(scores, w2, b2)
        scores, fc_2_cache = affine_forward(scores, w3, b3)

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dout = softmax_loss(scores, y)

        # Do the backpropagation:
        dx3, dw3, db3 = affine_backward(dout, fc_2_cache)
        dx2, dw2, db2 = affine_relu_backward(dx3, fc_1_cache)
        dx1, dw1, db1 = conv_relu_pool_backward(dx2, cv_cache)

        # L2 regularization:
        w1 += self.reg * w1
        w2 += self.reg * w2
        w3 += self.reg * w3

        grads["W3"], grads["W2"], grads["W1"] = dw3, dw2, dw1
        grads["b3"], grads["b2"], grads["b1"] = b3, b2, b1

        return loss, grads
