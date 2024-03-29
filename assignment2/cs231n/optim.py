import numpy as np

"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the momentum update formula. Store the updated value in #
    # the next_w variable. You should also use and update the velocity v.     #
    ###########################################################################
    v = config["momentum"] * v - config["learning_rate"] * dw
    next_w = w + v

    config["velocity"] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of w #
    # in the next_w variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Blueprint as given in course notes:
    #   cache = decay_rate * cache + (1 - decay_rate) * dx ** 2
    #   x += - learning_rate * dx / (np.sqrt(cache) + eps)

    # Extract convenience vars:
    learning_rate, decay_rate, epsilon, cache = (
        config["learning_rate"],
        config["decay_rate"],
        config["epsilon"],
        config["cache"],
    )

    # Do the actual work:
    cache = decay_rate * cache + (1 - decay_rate) * dw ** 2
    w -= learning_rate * dw / (np.sqrt(cache) + epsilon)

    # Actually update the next_w var:
    next_w = w

    # Repack convenience vars into our config:
    config["learning_rate"] = learning_rate
    config["decay_rate"] = decay_rate
    config["epsilon"] = epsilon
    config["cache"] = cache

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    next_w = None
    ###########################################################################
    # TODO: Implement the Adam update formula, storing the next value of w in #
    # the next_w variable. Don't forget to update the m, v, and t variables   #
    # stored in config.                                                       #
    #                                                                         #
    # NOTE: In order to match the reference output, please modify t _before_  #
    # using it in any calculations.                                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # # t is your iteration counter going from 1 to infinity
    # m = beta1 * m + (1 - beta1) * dx
    # mt = m / (1 - beta1 ** t)
    # v = beta2 * v + (1 - beta2) * (dx ** 2)
    # vt = v / (1 - beta2 ** t)
    # x += - learning_rate * mt / (np.sqrt(vt) + eps)

    # Extract convenience vars from config:
    t, m, v, epsilon, beta_1, beta_2, learning_rate = (
        config["t"],
        config["m"],
        config["v"],
        config["epsilon"],
        config["beta1"],
        config["beta2"],
        config["learning_rate"],
    )
    # Since default t is zero, we infer that iteration occurs before the work.
    t += 1

    # Do the actual work:
    m = beta_1 * m + (1 - beta_1) * dw
    mt = m / (1 - beta_1 ** t)
    v = beta_2 * v + (1 - beta_2) * (dw ** 2)
    vt = v / (1 - beta_2 ** t)
    w -= learning_rate * mt / (np.sqrt(vt) + epsilon)

    # NOTE: Like before didn't actually update next_w up there... record it here for now.
    next_w = w

    # Update the config with any updated convenience var values:
    config["t"] = t
    config["m"] = m
    config["v"] = v
    config["epsilon"] = epsilon
    config["beta1"] = beta_1
    config["beta2"] = beta_2
    config["learning_rate"] = learning_rate

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config
