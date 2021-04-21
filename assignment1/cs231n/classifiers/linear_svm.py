from builtins import range
import numpy as np
from random import shuffle  # noqa
from past.builtins import xrange  # noqa


# noinspection PyPep8Naming
def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    # dW: np.ndarray = np.zeros(W.shape)  # initialize the gradient as zero
    # # compute the loss:
    # num_classes = W.shape[1]
    # num_train = X.shape[0]
    # loss = 0.0
    # for i in range(num_train):
    #     # (hinge) loss -> gradient is 0/1 loss.
    #     scores = X[i].dot(W)
    #     correct_class_score = scores[y[i]]
    #     for j in range(num_classes):
    #         if j == y[i]:
    #             if margin < 0:
    #                 dW.transpose()[j] += -X[i]
    #             continue
    #         margin = scores[j] - correct_class_score + 1
    #         if margin > 0:
    #             loss += margin
    #             # vanilla saves into _rows_ of dW, transpose saves into _Cols_
    #             dW.transpose()[j] += X[i]
    #         else:
    #             loss += 0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                # Accumulate change due to derivative w.r.t. correct class:
                # dW[:, y[i]] = dW[:, y[i]] - X[i]
                dW[:, y[i]] -= X[i]
                # Accumulate change due to derivative w.r.t other classes:
                dW[:, j] += X[i]  # (Globally) acumulate change.

                # Honestly, I still can't tell whether we take the derivative
                # with respect to only 'other' classes j!=y[i]
                #
                # Or (as my hunch suspects) is that we take the derivative
                # with respect to both y_i and ALL j.
                #
                # I know that this code works, I'm just not positive why
                # or how the math checks out.

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW = dW / num_train

    # Add regularization to the loss & grad (vectorized).
    loss += reg * np.sum(W * W)
    dW = dW + reg * 2 * W

    ############################################################################
    # TODO:                                                                    #
    # Compute the gradient of the loss function and store it dW.               #
    # Rather than first computing the loss and then computing the derivative,  #
    # it may be simpler to compute the derivative at the same time that the    #
    # loss is being computed. As a result you may need to modify some of the   #
    # code above to compute the gradient.                                      #
    ############################################################################
    # They want me to do https://cs231n.github.io/optimization-1/#analytic
    #   when you’re implementing this in code you’d simply count the num classes
    #   which don't meet the desired margin (i.e. contributed to loss)
    #   and then the data vector x_i scaled by this number is the gradient.

    # Per the docs:
    # https://numpy.org/doc/stable/reference/generated/numpy.gradient.html
    # by default, it computes gradient with axis=0 AND axis=1,
    # returning both. We only want one of them. Test both to see what's right.

    return loss, dW


# noinspection PyPep8Naming
def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    ############################################################################
    # TODO:                                                                    #
    # Implement a vectorized version of the structured SVM loss, storing the   #
    # result in loss.                                                          #
    ############################################################################
    num_train = X.shape[0]
    scores = X.dot(W)
    correct_class_scores = scores[np.arange(num_train), y].reshape(num_train, 1)
    margin = np.maximum(0, scores - correct_class_scores + 1)
    margin[np.arange(num_train), y] = 0  # Negate any changes to correct class.
    loss = margin.sum() / num_train
    # Throws a few RuntimeWarning: overflows, suppress like this:
    with np.errstate(over="ignore"):
        loss += reg * np.sum(W * W)  # Add regularization term to loss.

    ############################################################################
    # TODO:                                                                    #
    # Implement a vectorized version of the gradient for the structured SVM    #
    # loss, storing the result in dW.                                          #
    #                                                                          #
    # Hint: Instead of computing the gradient from scratch, it may be easier   #
    # to reuse some of the intermediate values that you used to compute the    #
    # loss.                                                                    #
    ############################################################################
    # Compute gradient
    margin[margin > 0] = 1
    sum_loss_occurrences = margin.sum(axis=1)
    # Negate any changes to correct class.
    margin[np.arange(num_train), y] -= sum_loss_occurrences
    dW = X.T.dot(margin) / num_train  # Accumulate change & average it.
    dW = dW + reg * 2 * W  # Include regularization term.

    return loss, dW
