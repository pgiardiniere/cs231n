# Run some setup code for this notebook.
# import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

# Local imports:
from cs231n.classifiers import KNearestNeighbor


# This is some magic to make matplotlib figures appear inline in the notebook
# rather than in a new window.
# %matplotlib inline
# plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'
# Some more magic so that the notebook will reload external python modules:
# http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# %load_ext autoreload
# %autoreload 2

# Load the raw CIFAR-10 data.
cifar10_dir = "cs231n/datasets/cifar-10-batches-py"

if True:  # Pass cell 2
    # # Cleaning up variables to prevent loading data multiple times
    # (which may otherwise cause memory issue)
    # try:
    #    del X_train, y_train
    #    del X_test, y_test
    #    print('Clear previously loaded data.')
    # except:
    #    pass
    pass

X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# black's autoformatting. It works!
variable1, variable2, variable3, variable4, variable5, variable6 = (
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
)

# # As a sanity check, we print out the size of the training and test data.
# print("Training data shape: ", X_train.shape)
# print("Training labels shape: ", y_train.shape)
# print("Test data shape: ", X_test.shape)
# print("Test labels shape: ", y_test.shape)

if True:  # Pass cell 4
    # Visualize some examples from the dataset.
    # We show a few examples of training images from each class.
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
    num_classes = len(classes)
    samples_per_class = 7

    for i in range(3):
        for y, cls in enumerate(classes):
            idxs = np.flatnonzero(y_train == y)
            idxs = np.random.choice(idxs, samples_per_class, replace=False)
            for i, idx in enumerate(idxs):
                plt_idx = i * num_classes + y + 1
                plt.subplot(samples_per_class, num_classes, plt_idx)
                plt.imshow(X_train[idx].astype('uint8'))
                plt.axis('off')
                if i == 0:
                    plt.title(cls)
        plt.show()
    pass

# Subsample the data for more efficient code execution in this exercise
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]
num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

if True:  # Basic data prints
    # # Pete debug prints: Basic info.
    # print("--- X_train ---")
    # print(f"X_train shape: {X_train.shape}")
    # print(f"X_train:\n{X_train}")

    # print("--- X_test ---")
    # print(f"X_test shape: {X_test.shape}")
    # print(f"X_test:\n{X_test}")

    # print(f"X_train max: {np.max(X_train)}")
    # print(f"X_train min: {np.min(X_train)}")
    # print(f"X_train dtype: {X_train.dtype}")
    pass

# Create a kNN classifier instance.
# Remember that training a kNN classifier is a noop:
# the Classifier simply remembers the data and does no further processing
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)


# TODO: Open cs231n/classifiers/k_nearest_neighbor.py and implement
#   compute_distances_two_loops.

# Test your implementation:
dists = classifier.compute_distances_two_loops(X_test)


# TODO: Now implement the function predict_labels and run the code below:
# We use k = 1 (which is Nearest Neighbor).
y_test_pred = classifier.predict_labels(dists, k=1)
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print(f"Got {num_correct} / {num_test} correct => accuracy: {accuracy}")
# Expected accuracy is ~27%

# Now use k = 5.
y_test_pred = classifier.predict_labels(dists, k=5)
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print(f"Got {num_correct} / {num_test} correct => accuracy: {accuracy}")
# Expected accuracy is "marginally better" than k=1.
