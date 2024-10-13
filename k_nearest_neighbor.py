import numpy as np

class KNearestNeighbor:

    def __init__(self):
        pass

    def fit(self, X, y):

        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):

        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i, j] = np.sqrt(np.sum((X[i] - self.X_train[j])**2))
        return dists

    def compute_distances_one_loop(self, X):

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i, :] = np.sqrt(np.sum((X[i] - self.X_train)**2, axis=1))
        return dists

    def compute_distances_no_loops(self, X):

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        # Compute squared differences and sum along the last axis
        QQ = np.sum(self.X_train**2, axis=1)[:, np.newaxis]
        PP = np.sum(X**2, axis=1)[np.newaxis, :]
        QP = self.X_train.dot(X.T)

        dists = np.sqrt(QQ - 2 * QP + PP)
        return dists.T

    def predict_labels(self, dists, k=1):

        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            nearest_neighbors_indices = np.argsort(dists[i, :])[:k]
            closest_y = self.y_train[nearest_neighbors_indices]
            y_pred[i] = np.argmax(np.bincount(closest_y))

        return y_pred
