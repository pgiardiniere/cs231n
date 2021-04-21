from builtins import object
from typing import Union

import numpy as np

# from cs231n.classifiers.linear_svm import svm_loss_naive
from cs231n.classifiers.linear_svm import svm_loss_vectorized

# from cs231n.classifiers.softmax import softmax_loss_naive,
from cs231n.classifiers.softmax import softmax_loss_vectorized


# noinspection PyPep8Naming
class LinearClassifier(object):
    def __init__(self):
        self.W: Union[np.ndarray, None] = None

    def train(
        self,
        X,
        y,
        learning_rate=1e-3,
        reg=1e-5,
        num_iters=100,
        batch_size=200,
        verbose=False,
    ):
        """
        Train this linear classifier using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training
        iteration.
        """
        num_train, dim = X.shape
        # assume y takes values 0...K-1 where K is number of classes
        num_classes = np.max(y) + 1
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            ####################################################################
            # TODO:                                                            #
            # Sample batch_size elements from the training data and their      #
            # corresponding labels to use in this round of gradient descent.   #
            # Store the data in X_batch and the labels in (reltaed) y_batch.   #
            # after sampling X_batch should have shape (batch_size, dim)       #
            # and y_batch should have shape (batch_size,)                      #
            #                                                                  #
            # Hint: Use np.random.choice to generate indices. Sampling with    #
            # replacement is faster than sampling without replacement.         #
            #                                                                  #
            # My Note: Sampling without replacement is not just faster from a  #
            # purely computational perspective - it converges faster!          #
            # https://stats.stackexchange.com/questions/235844/                #
            ####################################################################
            indices = np.random.choice(
                a=np.arange(y.size), size=batch_size, replace=False
            )
            X_batch = X[indices, :]
            y_batch = y[indices]

            # evaluate loss and gradient (author's code, do not modify)
            # noinspection PyTupleAssignmentBalance
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # Update the weights using the gradient and the learning rate.
            self.W -= learning_rate * grad

            # print some intermediate output if verbose execution is on:
            if verbose and it % 100 == 0:
                print(f"iteration {it} / {num_iters}: loss {loss:.4f}")

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        y_pred = np.argmax(np.dot(X, self.W), axis=1)

        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
        data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for
        the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        # print("this method will never run without an override")
        pass


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
