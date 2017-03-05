import numpy as np

from deepforest.utils import check_models


class Layer(object):
    """
    A collection of models that may be chained to other layers in order to
    build a full deep forest model.
    """

    def __init__(self, layer, *models):
        self.parent_layer = layer
        self.models = check_models(models)

    def fit(self, X, y):
        self.parent_layer.fit(X, y)
        self._fit_internal_models(X, y)

    def _fit_internal_models(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = [model.predict_proba(X) for model in self.models]
        return np.concatenate(predictions, axis=1)


class InputLayer(Layer):
    """
    A layer that is not built upon another layer.
    """

    def __init__(self, *models):
        super(InputLayer, self).__init__(None, *models)

    def fit(self, X, y):
        self._fit_internal_models(X, y)
