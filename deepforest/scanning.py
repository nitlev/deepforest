import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d

from deepforest.models import Models


class Scanning(object):
    def __init__(self, models, n_patch, patch_size=(2, 2)):
        self.models = Models(models)
        self.n_patch = n_patch
        self.patch_size = patch_size

    def fit(self, X, y):
        transformed_X, transformed_y = self._scan(X, y)
        self.models.fit(transformed_X, transformed_y)

    def transform(self, X, y):
        transformed_X, transformed_y = self._scan(X, y)
        models_predictions = self.models.predict_proba(transformed_X)
        return models_predictions.reshape(len(X), -1)

    def fit_transform(self, X, y):
        transformed_X, transformed_y = self._scan(X, y)
        self.models.fit(transformed_X, transformed_y)
        models_predictions = self.models.predict_proba(transformed_X)
        return models_predictions.reshape(len(X), -1)

    def _scan(self, X, y=None):
        arrays, n_patches = zip(*(self._patch(x) for x in X))
        new_X = np.vstack(arrays)

        if y is not None:
            new_y = np.repeat(y, n_patches, axis=0)
            return new_X, new_y
        else:
            return new_X

    def _patch(self, X):
        patches = extract_patches_2d(X, self.patch_size,
                                     max_patches=self.n_patch)
        lines = (np.hstack(np.hstack(x)) for x in patches)
        array = np.vstack(x for x in lines)
        return array, len(patches)
