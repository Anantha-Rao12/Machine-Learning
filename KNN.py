import numpy as np
from scipy.spatial import distance
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Measure straight-line distance between two vectors"""
    return distance.euclidean(a, b)


class simpleKNN():
    """A simple 1-Nearest Neighbor """

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Trains the Model on given X_train and Y_train"""
        self.X_train = X_train
        self.y_train = y_train

    def closest(self, row: list) -> None:
        """Find the closest data point to a new data point"""
        best_dist = euclidean_distance(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euclidean_distance(row, self.X_train[i])
            if dist < best_dist:
                # Update label if we find a closer point
                best_dist = dist
                best_index = i
        return self.y_train[best_index]

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Receives features and returns the most-likely output"""
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

# We load the iris data set to test our classifier


iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.5,
                                                    random_state=100)

my_classifier = simpleKNN()
my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)


acc = accuracy_score(y_test, predictions)

print(acc)  # with random state 100, we get a 97% accuracy

