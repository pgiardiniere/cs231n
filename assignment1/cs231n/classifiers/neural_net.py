from typing import Dict, Union

# import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

foo: str = "bar"
"""This is a docstring for foo.

foo is a string that equals bar.
"""


# noinspection PyPep8Naming
class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    params: Dict[str, Union[float, None, ndarray]]

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {
            "W1": std * np.random.randn(input_size, hidden_size),
            "b1": np.zeros(hidden_size),
            "W2": std * np.random.randn(hidden_size, output_size),
            "b2": np.zeros(output_size),
        }

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        N, D = X.shape

        ########################################################################
        # TODO: Perform the forward pass, computing class scores for the input.#
        # Store the result in the scores variable, which should be an array of #
        # shape (N, C).                                                        #
        ########################################################################
        # fully-connected hidden layer 1 scores computation.
        hidden_1 = np.matmul(X, W1) + b1
        # X2 is the input matrix for hidden layer two of shape
        # (N, H); where N=num samples & H=num of hidden layer neurons
        #
        # Calculate X2 using ReLU (Rectified Linear Unit) activation function.
        # This is an element-wise (i.e. neuron-level) function
        # where intermediate scores <= 0 get dropped (set to zero).
        X2 = np.maximum(0, hidden_1)
        # fully-connected output layer is simply the final scores
        # scores has shape (C, N)
        scores = np.matmul(X2, W2) + b2

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, **which should be a scalar**. Use the Softmax       #
        # classifier loss.                                                          #
        #                                                                           #
        # ** REQUIRED: loss is NOT stored as a scalar in this assignment!!!         #
        # fix your documentation dudes.                                             #
        #############################################################################
        # Using scores from above, control exponents' values for numerical stability
        stab_scores = scores - np.max(scores, axis=1, keepdims=True)
        softmaxes = np.exp(stab_scores) / np.sum(
            np.exp(stab_scores), axis=1, keepdims=True
        )
        loss = np.sum(-np.log(softmaxes[np.arange(N), y]))
        # Take average on the cumulative loss
        loss /= N
        # Add regularization term. Account for both layers!
        loss += reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2))

        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        # Again, must use a clever trick and subtract out -1 term on derivative
        # with slicing.
        #
        # Also, average each entry by total num samples, so that the row-wise sums
        # we use will also constitute an average value.
        softmaxes[np.arange(N), y] -= 1
        softmaxes /= N

        # Backwards pass begins at end (duh), dW2 & db2
        dW2 = np.matmul(X2.T, softmaxes)
        db2 = softmaxes.sum(axis=0)

        # W1, b1  (derivative of hidden_1 required to compute)
        dW1 = np.matmul(softmaxes, W2.T)
        d_hidden_1 = dW1 * (hidden_1 > 0)  # zero-out negative scores
        dW1 = np.matmul(X.T, d_hidden_1)
        db1 = d_hidden_1.sum(axis=0)

        # Accumulate the derivative of regularization term for each grad:
        dW1 += 2 * reg * W1
        dW2 += 2 * reg * W2

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

        return loss, grads

    def train(
        self,
        X,
        y,
        X_val,
        y_val,
        learning_rate=1e-3,
        learning_rate_decay=0.95,
        reg=5e-6,
        num_iters=100,
        batch_size=200,
        verbose=False,
    ):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            indices = np.random.randint(low=0, high=X.shape[0], size=batch_size)
            X_batch = X[indices, :]
            y_batch = y[indices]
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            for weights_or_bias in self.params:
                self.params[weights_or_bias] -= (
                    learning_rate * grads[weights_or_bias]
                )
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            "loss_history": loss_history,
            "train_acc_history": train_acc_history,
            "val_acc_history": val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        # Nevermind, just had to ignore the docs which were lying
        # as opposed to the prior assignments. Nbd
        y_pred = np.argmax(self.loss(X), axis=1)

        return y_pred
