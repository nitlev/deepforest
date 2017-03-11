import numpy as np

from deepforest.models import Models
from deepforest.utils import check_models


class Layer(object):
    """
    A collection of models that may be chained to other layers in order to
    build a full deep forest model.
    """

    def __init__(self, layer, *models):
        self.parent_layer = layer
        self.models = Models(check_models(models))

    def fit(self, X, y):
        self.parent_layer.fit(X, y)
        full_X = self._add_parent_predictions(X)
        self.models.fit(full_X, y)
        return self

    def _add_parent_predictions(self, X):
        predictions = self.parent_layer.predict(X)
        if len(predictions.shape) == 1:
            new_X = np.concatenate([X, np.reshape(predictions, (-1, 1))],
                                   axis=1)
        else:
            new_X = np.concatenate([X, predictions], axis=1)
        return new_X

    def predict(self, X):
        full_X = self._add_parent_predictions(X)
        return self.models.predict_proba(full_X)


class InputLayer(object):
    """
    A layer that is not built upon another layer.
    """

    def __init__(self, *models):
        self.models = Models(check_models(models))

    def fit(self, X, y):
        self.models.fit(X, y)
        return self

    def predict(self, X):
        return self.models.predict(X)
