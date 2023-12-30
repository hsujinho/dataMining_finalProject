import numpy as np
import pandas as pd

# implementation of decision tree regressor
class node:
    def __init__(self, value, left, right, attribute, threshold):
        self.value = value  # mean of y values
        self.left = left  # left subtree
        self.right = right  # right subtree
        self.attribute = attribute  # attribute to split on
        self.threshold = threshold  # threshold value to split on

# decision tree regressor class
class decisionTreeRegressor:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y
        self.tree = self.build_tree(X_np, y_np, depth=0)

    # find best split for a node
    def find_best_split(self, X, y):
        m, n = X.shape
        # if only one data point, return None
        if m <= 1:
            return None, None

        best_attribute = None
        best_threshold = None
        best_loss = np.inf
        # iterate through all attributes
        for att in range(n):
            # iterate through all possible thresholds
            thresholds = np.unique(X[:, att])
            for threshold in thresholds:
                # split data
                left = X[:, att] <= threshold
                right = X[:, att] > threshold
                y_left = y[left]
                y_right = y[right]
                # if no split, skip
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                # calculate mean squared error
                MSE_left = np.mean((y[left] - np.mean(y[left])) ** 2)
                MSE_right = np.mean((y[right] - np.mean(y[right])) ** 2)
                # update best split if total MSE is lower
                if MSE_left + MSE_right < best_loss:
                    best_attribute = att
                    best_threshold = threshold
                    best_loss = MSE_left + MSE_right

        return best_attribute, best_threshold
        # build tree recursively

    def build_tree(self, X, y, depth):
        # find best split
        best_attribute, best_threshold = self.find_best_split(X, y)
        # if no split or max depth reached, return leaf node
        if best_attribute is None or depth == self.max_depth:
            leaf_value = np.mean(y)
            return node(leaf_value, None, None, best_attribute, best_threshold)
        # if split exists, build tree recursively
        left = X[:, best_attribute] <= best_threshold
        right = X[:, best_attribute] > best_threshold
        left_child = self.build_tree(X[left], y[left], depth + 1)
        right_child = self.build_tree(X[right], y[right], depth + 1)
        return node(np.mean(y), left_child, right_child, best_attribute, best_threshold)

    # predict instance recursively
    def predict_instance(self, x, tree):
        # if is leaf node, return value
        if tree.left is None and tree.right is None:
            return tree.value
        # if it is not leaf node, compare attribute value to threshold and go left or right
        if x[tree.attribute] <= tree.threshold:
            return self.predict_instance(x, tree.left)
        return self.predict_instance(x, tree.right)

    # predict dataset
    def predict(self, X):
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        predictions = []
        for x in X_np:
            predictions.append(self.predict_instance(x, self.tree))
        return np.array(predictions)
