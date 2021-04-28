from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # Hidden layer is W1,b1. Output layer is W2,b2
        #   W1 shape: input_dim, hidden_dim
        #   b1 shape: hidden_dim
        #   W2 shape: hidden_dim, num_classes
        #   b2 shape: num_classes
        self.params = {
            "W1": np.random.randn(input_dim, hidden_dim) * weight_scale,
            "b1": np.zeros(hidden_dim),
            "W2": np.random.randn(hidden_dim, num_classes) * weight_scale,
            "b2": np.zeros(num_classes),
        }
        self.reg = reg

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:

        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:

        If y is None, then run a test-time forward pass of the model and return:

        - scores: Array of shape (N, C) giving classification scores, where scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:

        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter names to gradients of the loss with respect to those parameters.
        """
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        scores = None
        w1, b1, w2, b2 = self.params.values()

        # Compute raw fc-net scores: Affine -> relu -> affine
        scores, cache_1 = affine_relu_forward(X, w1, b1)
        scores, cache_2 = affine_forward(scores, w2, b2)

        # If y is None then we are in test mode so just return scores.
        if y is None:
            return scores

        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss = 0.0
        # Compute fc-net loss: Affine -> relu -> affine -> softmax
        # Include an L2 regularization term
        loss, d_softmax = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(w1 ** 2) + np.sum(w2 ** 2))

        dx2, dw2, db2 = affine_backward(d_softmax, cache_2)
        dx1, dw1, db1 = affine_relu_backward(dx2, cache_1)
        dw2 += self.reg * w2
        dw1 += self.reg * w1

        grads = {
            "W1": dw1,
            "b1": db1,
            "W2": dw2,
            "b2": db2,
        }

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        #####################################################################x#######
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # Stack layer dimensionalities in the order we process them.
        #
        # Then, walk through the layers, from: Input > Hidden_1 > ... > Hidden_n > Output,
        # creating Weights & Biases as we go.
        layer_dims = np.hstack([input_dim, hidden_dims, num_classes])
        for i in range(self.num_layers):
            self.params[f"W{i+1}"] = (
                np.random.randn(layer_dims[i], layer_dims[i + 1]) * weight_scale
            )
            self.params["b" + f"{i+1}"] = np.zeros(layer_dims[i + 1])

        if self.normalization is not None:
            for i in range(self.num_layers - 1):
                self.params[f"beta{i + 1}"] = np.zeros(layer_dims[i + 1])
                self.params[f"gamma{i + 1}"] = np.ones(layer_dims[i + 1])

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        # if self.use_dropout:  # TRIAL: use_dropout is superfluous
        self.dropout_param = {"mode": "train", "p": dropout}
        if seed is not None:
            self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        # if self.use_dropout:  # TRIAL: self.use_dropout is superfluous.
        self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None

        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        fc_caches = [*np.zeros(self.num_layers)]
        relu_caches = [*np.zeros(self.num_layers - 1)]
        do_caches = [*np.zeros(self.num_layers - 1)]
        bn_caches = [*np.zeros(self.num_layers - 1)]
        beta = gamma = bn_param = None

        # For all layers, compute the forward pass. Layers follow the given pattern:
        # affine -> batchnorm or layernorm -> relu -> dropout
        #
        # Normalization & dropout are optional. Special cases at first/last entries.
        for i in range(self.num_layers):
            W = self.params[f"W{i + 1}"]
            b = self.params[f"b{i + 1}"]
            if self.normalization is not None and i < self.num_layers - 1:
                beta = self.params[f"beta{i + 1}"]
                gamma = self.params[f"gamma{i + 1}"]
                bn_param = self.bn_params[i]

            if i == 0:
                scores, fc_caches[i] = affine_forward(X, W, b)
                scores, bn_caches[i] = self.norm_forward(scores, gamma, beta, bn_param)
                scores, relu_caches[i] = relu_forward(scores)
                scores, do_caches[i] = dropout_forward(scores, self.dropout_param)
            elif i < self.num_layers - 1:
                scores, fc_caches[i] = affine_forward(scores, W, b)
                scores, bn_caches[i] = self.norm_forward(scores, gamma, beta, bn_param)
                scores, relu_caches[i] = relu_forward(scores)
                scores, do_caches[i] = dropout_forward(scores, self.dropout_param)
            else:
                scores, fc_caches[i] = affine_forward(scores, W, b)

        # If test mode return early
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize       #
        # the scale and shift parameters.                                          #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, d_softmax = softmax_loss(scores, y)

        regularizer_sum = 0
        for i in range(self.num_layers):
            regularizer_sum += np.sum(self.params[f"W{i+1}"] ** 2)
        loss += 0.5 * self.reg * regularizer_sum

        # Backward pass - Calculate the grads similar to scores above
        dXs = [*np.zeros(self.num_layers)]
        dWs = [*np.zeros(self.num_layers)]
        dbs = [*np.zeros(self.num_layers)]
        dgammas = [*np.zeros(self.num_layers)]
        dbetas = [*np.zeros(self.num_layers)]

        # For all layers, compute the backward pass. Special case at the first iteration.
        # Iterate backwards through the layers, creating the grads as we go.
        for i in reversed(range(self.num_layers)):
            if i == self.num_layers - 1:
                dXs[i], dWs[i], dbs[i] = affine_backward(d_softmax, fc_caches[i])
            else:
                dx = dropout_backward(dXs[i + 1], do_caches[i])
                dx = relu_backward(dx, relu_caches[i])
                dx, dgammas[i], dbetas[i] = self.norm_backward(dx, bn_caches[i])
                dXs[i], dWs[i], dbs[i] = affine_backward(dx, fc_caches[i])

        for i, dW in enumerate(dWs):
            # Include l2 regularization term on weights gradients
            dW += self.reg * self.params[f"W{i + 1}"]

            grads[f"W{i+1}"] = dWs[i]
            grads[f"b{i+1}"] = dbs[i]
            if self.normalization is not None and i < len(dWs) - 1:
                grads[f"gamma{i+1}"] = dgammas[i]
                grads[f"beta{i+1}"] = dbetas[i]

        return loss, grads

    def norm_forward(self, x: np.ndarray, gamma, beta, bn_param):
        """
        Wrapper function which calls the correct batch or layer norm, if any.
        """
        if bn_param is None:
            return x, None
        elif self.normalization == "batchnorm":
            scores, bn_cache = batchnorm_forward(x, gamma, beta, bn_param)
            return scores, bn_cache
        elif self.normalization == "layernorm":
            pass

    def norm_backward(self, dx: np.ndarray, cache):
        """
        Wrapper function which calls the correct batch or layer norm, if any.
        """
        if self.normalization is None:
            return dx, None, None
        elif self.normalization == "batchnorm":
            dx, dgamma, dbeta = batchnorm_backward_alt(dx, cache)
            return dx, dgamma, dbeta
        elif self.normalization == "layernorm":
            pass
