import pandas as pd

from deepforest.utils import check_models, predictions_to_dataframe


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
        full_X = self._add_parent_predictions(X)
        return self._fit_internal_models(full_X, y)

    def _add_parent_predictions(self, X):
        predictions = self.parent_layer.predict(X)
        new_X = pd.concat([X, predictions], axis=1)
        return new_X

    def _fit_internal_models(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        full_X = self._add_parent_predictions(X)
        return self._make_predictions(full_X)

    def _make_predictions(self, X):
        predictions = [model.predict_proba(X) for model in self.models]
        return predictions_to_dataframe(predictions, X.index, len(self.models))


class InputLayer(Layer):
    """
    A layer that is not built upon another layer.
    """

    def __init__(self, *models):
        super(InputLayer, self).__init__(None, *models)

    def fit(self, X, y):
        return self._fit_internal_models(X, y)

    def predict(self, X):
        return self._make_predictions(X)
