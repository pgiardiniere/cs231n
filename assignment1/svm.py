import math
import time
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from cs231n.classifiers import LinearSVM
from cs231n.classifiers.linear_svm import svm_loss_naive, svm_loss_vectorized
from cs231n.data_utils import load_CIFAR10
from cs231n.gradient_check import grad_check_sparse


# CIFAR-10 Data Loading and Preprocessing
cifar10_dir = "cs231n/datasets/cifar-10-batches-py"
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# # Visualize some examples from the dataset.
# # We show a few examples of training images from each class.
# classes = [
#     "plane",
#     "car",
#     "bird",
#     "cat",
#     "deer",
#     "dog",
#     "frog",
#     "horse",
#     "ship",
#     "truck",
# ]
# num_classes = len(classes)
# samples_per_class = 7
# for y, cls in enumerate(classes):
#     idxs = np.flatnonzero(y_train == y)
#     idxs = np.random.choice(idxs, samples_per_class, replace=False)
#     for i, idx in enumerate(idxs):
#         plt_idx = i * num_classes + y + 1
#         plt.subplot(samples_per_class, num_classes, plt_idx)
#         plt.imshow(X_train[idx].astype("uint8"))
#         plt.axis("off")
#         if i == 0:
#             plt.title(cls)
# plt.show()
pass

# Split the data into train, val, and test sets. In addition we will
# create a small development set as a subset of the training data;
# we can use this for development so our code runs faster.
num_training = 49000
num_validation = 1000
num_test = 1000
num_dev = 500

# Our validation set will be num_validation points from the original
# training set.
mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask]

# Our training set will be the first num_train points from the original
# training set.
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

# We will also make a development set, which is a small subset of
# the training set.
mask = np.random.choice(num_training, num_dev, replace=False)
X_dev = X_train[mask]
y_dev = y_train[mask]

# We use the first num_test points of the original test set as our
# test set.
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# Preprocessing: reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

# # As a sanity check, print out the shapes of the data
# print("Training data shape: ", X_train.shape)
# print("Validation data shape: ", X_val.shape)
# print("Test data shape: ", X_test.shape)
# print("dev data shape: ", X_dev.shape)


# Preprocessing: subtract the mean image
# first: compute the image mean based on the training data
mean_image = np.mean(X_train, axis=0)

# plt.figure(figsize=(4, 4))
# plt.imshow(
#     mean_image.reshape((32, 32, 3)).astype("uint8")
# )  # visualize the mean image
# plt.show()
pass

# second: subtract the mean image from train and test data
X_train -= mean_image
X_val -= mean_image
X_test -= mean_image
X_dev -= mean_image

# third: append the bias dimension of ones (i.e. bias trick) so that our SVM
# only has to worry about optimizing a single weight matrix W.
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])


# SVM Classifier
#
# Your code for this section will all be written inside
# `cs231n/classifiers/linear_svm.py`.
#
# As you can see, we have prefilled the function
# `svm_loss_naive` which uses for loops to evaluate
# the multiclass SVM loss function.

# Evaluate the naive implementation of the loss we provided for you:

# generate a random SVM weight matrix of small numbers
W = np.random.randn(3073, 10) * 0.0001
loss, _ = svm_loss_naive(W, X_dev, y_dev, 0.000005)
# print(f"loss: {loss:.2f}")


# The `grad` returned from the function above is right now all zero.
# Derive and implement the gradient for the SVM cost function
# and implement it inline inside the function `svm_loss_naive`.
#
# You will find it helpful to interleave your new code
# inside the existing function.
#
# Do this using the 'analytic' gradient .
#
# To check that you have correctly implemented the gradient correctly,
# you can numerically estimate the gradient of the loss function and compare
# the numeric estimate to the gradient that you computed.
#
# We have provided code that does this for you:

# # First: We perform this check _without_ a regularization term.
# _, grad = svm_loss_naive(W, X_dev, y_dev, 0.0)
# f = lambda w: svm_loss_naive(w, X_dev, y_dev, 0.0)[0]  # noqa
# _ = grad_check_sparse(f, W, grad)
#
# # Now do the gradient check with a non-zero regularization term.
# loss, grad = svm_loss_naive(W, X_dev, y_dev, 5e1)
# f = lambda w: svm_loss_naive(w, X_dev, y_dev, 5e1)[0]  # noqa
# _ = grad_check_sparse(f, W, grad)

# YEP, they're near identical. Turning off this code chunk.

# **Inline Question 1**
#
# It is possible that once in a while a dimension in the gradcheck will not
# match exactly.
# * What could such a discrepancy be caused by?
# * Is it a reason for concern?
# * What is a simple example in one dimension where a gradient check could fail?
# * How would change the margin affect of the frequency of this happening?
#
# *Hint: the SVM loss function is not strictly speaking differentiable*

pass
# TODO: Remove whole chunk.
# # For now only compute the loss (we will implement the gradient in a moment).
# tic = time.time()
# loss_naive, _ = svm_loss_naive(W, X_dev, y_dev, 0.000005)
# toc = time.time()
# print(f"Naive loss: {loss_naive} computed in {toc - tic}")
#
# tic = time.time()
# loss_vectorized, _ = svm_loss_vectorized(W, X_dev, y_dev, 0.000005)
# toc = time.time()
# print(f"Vectorized loss: {loss_vectorized} computed in {toc - tic}")
pass

# Next implement the function svm_loss_vectorized. (Also compute the
# loss function's gradient in a vectorized manner)
#
# The losses should match and your vectorized implementation should be faster.

# The naive implementation and the vectorized implementation loss/grads
# should match, but the vectorized version will be much faster.
tic = time.time()
loss_naive, grad_naive = svm_loss_naive(W, X_dev, y_dev, 0.000005)
toc = time.time()

v_tic = time.time()
loss_vectd, grad_vectd = svm_loss_vectorized(W, X_dev, y_dev, 0.000005)
v_toc = time.time()

# # The loss is a float -> compare by subtraction.
# # The gradient is a matrix -> use the Frobenius norm to compare them.
# difference = np.linalg.norm(grad_naive - grad_vectd, ord="fro")
# print(f"Naive loss: {loss_naive:.2f} computed in {toc - tic:.4f}")
# print(f"Vectorized loss: {loss_vectd:.2f} computed in {v_toc - v_tic:.4f}")
# print(f"loss difference: {loss_naive - loss_vectd:.2f}")
# print(f"gradient difference: {difference:.2f}")


# ### Stochastic Gradient Descent
#
# We now have vectorized and efficient expressions for the loss,
# the gradient, and our gradient matches the numerical gradient.
# We are therefore ready to do SGD to minimize the loss.
# Your code for this part will be written inside:
# `cs231n/classifiers/linear_classifier.py`.

# In the file linear_classifier.py, implement SGD in the function
# LinearClassifier.train() and then run it with the code below.
svm = LinearSVM()
loss_hist = svm.train(
    X_train,
    y_train,
    learning_rate=1e-7,
    reg=2.5e4,
    num_iters=1500,
    verbose=True,
)

# A useful debugging strategy is to plot the loss as a function of
# iteration number:
plt.plot(loss_hist)
plt.xlabel("Iteration number")
plt.ylabel("Loss value")
plt.show()


# Write the LinearSVM.predict function and evaluate the performance on both the
# training and validation set
y_train_pred = svm.predict(X_train)
print(f"training accuracy: {np.mean(y_train == y_train_pred)}")
y_val_pred = svm.predict(X_val)
print(f"validation accuracy: {np.mean(y_val == y_val_pred)}", end="\n\n")


# Use the validation set to tune hyperparameters (regularization strength and
# learning rate). You should experiment with different ranges for the learning
# rates and regularization strengths; if you are careful you should be able to
# get a classification accuracy of about 0.39 on the validation set.

# NOTE: you may see runtime/overflow warnings during hyper-parameter search.
# This may be caused by extreme values, and is not a bug.
################################################################################
# TODO:                                                                        #
# Write code that chooses the best hyperparameters by tuning on the validation #
# set. For each combination of hyperparameters, train a linear SVM on the      #
# training set, compute its accuracy on the training and validation sets, and  #
# store these numbers in the results dictionary. In addition, store the best   #
# validation accuracy in best_val and the LinearSVM object that achieves this  #
# accuracy in best_svm.                                                        #
#                                                                              #
# Hint: You should use a small value for num_iters as you develop your         #
# validation code so that the SVMs don't take much time to train; once you are #
# confident that your validation code works, you should rerun the validation   #
# code with a larger value for num_iters.                                      #
################################################################################
# Methodology:
# No sophisticated cross-validation - just use basic 3-way split:
# Here, accuracy is simply the fraction of correctly classified data points.

# results: A dictionary mapping from tuples of the form
# (learning_rate, regularization_strength) to tuples of the form
# (training_accuracy, validation_accuracy).
results = {}
# best_val: The highest validation_accuracy that we have seen so far.
best_val = -1.0
# best_svm: The LinearSVM object that achieved the highest validation rate.
best_svm: LinearSVM = LinearSVM()
# Reference rates/strenghts. You may or may not want to change these hyperparams
learning_rates = [1e-7, 5e-5]
regularization_strengths = [2.5e4, 5e4]

for learn_rate in learning_rates:
    for reg in regularization_strengths:
        svm = LinearSVM()
        svm.train(
            X_train,
            y_train,
            learning_rate=learn_rate,
            reg=reg,
            num_iters=400,
            verbose=False,
        )
        y_train_pred = svm.predict(X_train)
        y_val_pred = svm.predict(X_val)
        train_acc = np.mean(y_train_pred == y_train)
        val_acc = np.mean(y_val_pred == y_val)

        if val_acc > best_val:
            best_val = val_acc
            best_svm = svm

        results[(learn_rate, reg)] = (train_acc, val_acc)

# Print results.
for learn_rate, reg in sorted(results):
    train_accuracy, val_accuracy = results[(learn_rate, reg)]
    print(f"With learning rate: {learn_rate:.8f} and regularizer: {reg:.0f}")
    print(
        f"  Train accuracy: {train_accuracy:.3f} & Val accuracy: {val_accuracy:.3f}"
    )
print(f"\nBest validation accuracy achieved: {best_val:.3f}")


# Visualize the cross-validation results:
x_scatter = [math.log10(x[0]) for x in results]
y_scatter = [math.log10(x[1]) for x in results]
# plot training accuracy
marker_size = 100
colors = [results[x][0] for x in results]
plt.subplot(2, 1, 1)
plt.tight_layout(pad=3)
plt.scatter(x_scatter, y_scatter, marker_size, c=colors, cmap="coolwarm")
plt.colorbar()
plt.xlabel("log learning rate")
plt.ylabel("log regularization strength")
plt.title("CIFAR-10 training accuracy")
# plot validation accuracy
colors = [results[x][1] for x in results]  # default size of markers is 20
plt.subplot(2, 1, 2)
plt.scatter(x_scatter, y_scatter, marker_size, c=colors, cmap="coolwarm")
plt.colorbar()
plt.xlabel("log learning rate")
plt.ylabel("log regularization strength")
plt.title("CIFAR-10 validation accuracy")
plt.show()


# Evaluate the best svm on test set
y_test_pred = best_svm.predict(X_test)
test_accuracy = np.mean(y_test == y_test_pred)
print(f"linear SVM on raw pixels final test set accuracy: {test_accuracy}")

# Visualize the learned weights for each class.
# Depending on your choice of learning rate and regularization strength,
# these may or may not be nice to look at.
w = best_svm.W[:-1, :]  # strip out the bias
w = w.reshape(32, 32, 3, 10)
w_min, w_max = np.min(w), np.max(w)
classes = [
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
for i in range(10):
    plt.subplot(2, 5, i + 1)

    # Rescale the weights to be between 0 and 255
    wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
    plt.imshow(wimg.astype("uint8"))
    plt.axis("off")
    plt.title(classes[i])
plt.show()

# **Inline question 2**
#
# Describe what your visualized SVM weights look like,
# and offer a brief explanation for why they look they way that they do.
