from builtins import object
import numpy as np

# My imports:
from sklearn.metrics.pairwise import euclidean_distances
import operator


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """

        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)

        print(f"\npredicting labels... k={k}")
        dists_argsorted = np.argsort(dists, axis=1)
        dists_nearest_k = dists_argsorted[:, :k]

        if k == 1:
            y_pred = self.y_train[np.ravel(dists_nearest_k)]
            return y_pred

        else:
            # Recall: Labels are in range(10)
            # Make a simple counter to determine which label occurs
            # most frequently for each vector in test.
            # print(f"dists_nearest_k: (first 5)\n{dists_nearest_k[0:5, :]}")
            # print(f"{self.y_train[420]}, {self.y_train[3684]}, {self.y_train[4224]}")
            # print(f"self.y_train - nearest k mask:\n{self.y_train[dists_nearest_k]}")
            masked_nearest_labels = self.y_train[dists_nearest_k]
            print(masked_nearest_labels, end="\n\n")

            if True:  # Broken 'vectorized' code
                # # 1) attempt to use apply_along_axis, then realize it's complicated
                # #    and not working intuitively
                # print(
                #     np.apply_along_axis(
                #         np.bincount, axis=1, arr=masked_nearest_labels,
                #     )
                # )
                # print(np.apply_along_axis(mode, axis=1, arr=masked_nearest_labels))
                # # 2) Vectorized numpy way: Apply masks 1 thru 10, sum the vals, return the max
                # maximal_value_finder = np.zeros(
                #     shape=(masked_nearest_labels.shape[0], 10)
                # )
                # print(maximal_value_finder.shape)
                # labels_mask = np.array(range(10))
                # print(labels_mask)
                # for label in range(10):
                #     occurrences = np.sum(label == masked_nearest_labels)
                #     print(occurrences)
                pass

            # Eh, just use loops instead
            for i in range(np.shape(masked_nearest_labels)[0]):
                label_counts = dict([x, 0] for x in range(10))
                for j in range(k):
                    label = masked_nearest_labels[i, j]
                    label_counts[label] += 1
                y_pred[i] = max(
                    label_counts.items(), key=operator.itemgetter(1)
                )[0]
            return y_pred

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

                pass

                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # This whole Double-loop / Single-loop / No-loop is silly. Just use scipy.
        dists = euclidean_distances(X, self.X_train)

        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists
