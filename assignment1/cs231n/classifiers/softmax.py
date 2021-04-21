import numpy as np


# noinspection PyPep8Naming
def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    #########################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops. #
    # Store the loss in loss and the gradient in dW. If you are not careful #
    # here, it is easy to run into numeric instability. Don't forget the    #
    # regularization!                                                       #
    #########################################################################
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_classes = W.shape[1]
    num_samples = X.shape[0]

    # Get scores & control exponent values for numerical stability
    scores = np.matmul(X, W)
    stab_scores = scores - np.max(scores, axis=1, keepdims=True)

    for i in range(num_samples):
        softmax = np.exp(stab_scores[i]) / np.sum(np.exp(stab_scores[i]))
        loss += -np.log(softmax[y[i]])
        # Gradient
        for j in range(num_classes):
            if j == y[i]:
                dW[:, j] += X[i] * (softmax[j] - 1)
            else:
                dW[:, j] += X[i] * softmax[j]

    # Average the cumulative loss and gradient
    loss /= num_samples
    dW /= num_samples

    # Include the Regularization penalty on the loss and gradient
    loss += reg * np.sum(W ** 2)
    dW += reg * 2 * W

    return loss, dW


# noinspection PyPep8Naming
def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    ###########################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.#
    # Store the loss in loss and the gradient in dW. If you are not careful   #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                                         #
    ###########################################################################
    # Initialize the (cumulative) loss and gradient to zero.
    dW: np.ndarray
    num_samples = X.shape[0]

    # Get scores & control exponent values for numerical stability
    scores = np.matmul(X, W)
    stab_scores = scores - np.max(scores, axis=1, keepdims=True)

    softmaxes: np.ndarray = np.exp(stab_scores) / np.sum(
        np.exp(stab_scores), axis=1, keepdims=True
    )
    # Slice out the y_i'th (correct class) entry from each row of softmax:
    target_classifiers = softmaxes[np.arange(softmaxes.shape[0]), y]
    loss = np.sum(-np.log(target_classifiers))

    # Calculate gradient:
    # It's convoluted in numpy to do the kind of indexing op I want
    # (as it doesn't have anything exactly like .loc like pandas)
    #
    # As such, we'll resort to a 'clever' trick and distribute the -1
    # in the naive implementation.
    dW_softmaxes = softmaxes
    dW_softmaxes[np.arange(dW_softmaxes.shape[0]), y] -= 1
    dW = np.matmul(X.T, dW_softmaxes)

    # Average the cumulative loss and gradient
    loss /= num_samples
    dW /= num_samples

    # Include the Regularization penalty on the loss and gradient
    loss += reg * np.sum(W ** 2)
    dW += reg * 2 * W

    return loss, dW
