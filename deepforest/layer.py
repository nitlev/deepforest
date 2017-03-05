import numpy as np


def check_models(models):
    at_least_one_model = False
    for model in models:
        check_model(model)
        at_least_one_model = True
    assert at_least_one_model, "You must pass at least one model to a Layer"
    return models


def check_model(model):
    has_fit_method = hasattr(model, "fit")
    has_predict_method = hasattr(model, "predict")
    assert has_fit_method and has_predict_method, \
        "models should have fit and predict methods"


class Layer(object):
    """
    A collection of models that may be chained to other layers in order to
    build a full deep forest model.
    """

    def __init__(self, layer, *models):
        self.models = check_models(models)

    def fit(self, X, y):
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
