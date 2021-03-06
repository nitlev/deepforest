import numpy as np

from deepforest.layer import Layer, InputLayer


class CascadeForest(object):
    """
    A simple implementation of the gcForest algorithme developed by Zhi-Hua
    Zhou and Ji Feng
    """

    def __init__(self, model_generator, metric, objective='maximize'):
        self.model_generator = model_generator
        self.metric = metric
        self.levels = 0
        self.input_layer = None
        self.output_layer = None
        self.objective = objective

    def fit(self, X, y):
        self.output_layer.fit(X, y)

    def predict_proba(self, X):
        layer_prediction = self.output_layer.predict_proba(X)
        return np.mean(layer_prediction, axis=-1)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def grow(self, Xtrain, ytrain, Xtest, ytest):
        score = - np.inf
        new_layer = InputLayer(*next(self.model_generator))
        new_layer.fit(Xtrain, ytrain)
        predictions = np.mean(new_layer.predict_proba(Xtest), axis=-1)
        current_metric = self.metric(ytest, predictions[:, 1])

        while self._has_improved(current_metric, score):
            score = current_metric
            self._add_layer(new_layer)
            new_layer = self._new_layer()
            new_layer.fit(Xtrain, ytrain)
            predictions = np.mean(new_layer.predict_proba(Xtest), axis=-1)
            current_metric = self.metric(ytest, predictions[:, 1])

    def _new_layer(self):
        return Layer(self.output_layer,
                     *next(self.model_generator))

    def _add_layer(self, layer):
        self.levels += 1
        self.output_layer = layer

    def _has_improved(self, current_metric, last_score):
        if self.objective == 'maximize':
            return current_metric > last_score
        else:
            return current_metric < last_score
