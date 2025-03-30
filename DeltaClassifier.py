from enum import Enum
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import manhattan_distances, cosine_distances


class DeltaMeasure(Enum):
    BURROW = 'Burrow',
    COSINE = 'Cosine'


class DeltaClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, scale=True, delta=DeltaMeasure.BURROW):
        self.delta = delta
        self.scale = scale

    def fit(self, X, y):
        self.X_train = X
        self.y_train = np.array(y)
        self.classes_ = np.unique(y)
        self.scaler = None

        # Optional scaling of training data
        if self.scale:
            self.scaler = StandardScaler(with_mean=False)
            self.X_train = self.scaler.fit_transform(X)

        return self

    def predict(self, X):
        probabilities = self.predict_proba(X)
        predictions = np.array([self.classes_[np.argmax(probs)]
                               for probs in probabilities])
        return predictions

    def predict_proba(self, X):
        # Scale the test data if needed
        if self.scale and self.scaler is not None:
            X = self.scaler.transform(X)

        # Compute Manhattan distances between test points and all training points
        # Sparse-aware distance calculation
        if self.delta == DeltaMeasure.BURROW:
            distances = manhattan_distances(X, self.X_train)
        elif self.delta == DeltaMeasure.COSINE:
            distances = cosine_distances(X, self.X_train)

        # Convert distances into probabilities (smaller distance -> higher probability)
        probabilities = []
        for test_point_distances in distances:
            class_probabilities = []
            # Calculate probabilities for each class
            for label in self.classes_:
                # Get indices of training points with the current label
                class_indices = np.where(self.y_train == label)[0]
                # Get distances for test points with the current label
                class_distances = test_point_distances[class_indices]
                # Calculate minimum distance to all points with the current label
                min_distance = np.min(class_distances)
                # avg_distance = np.mean(class_distances)
                # Smaller avg = higher prob
                class_probabilities.append(1 / (min_distance + 1e-8))
            class_probabilities = np.array(class_probabilities)
            class_probabilities /= class_probabilities.sum()  # Normalize
            probabilities.append(class_probabilities)

        return np.array(probabilities)
