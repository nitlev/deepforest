import numpy as np
from sklearn import clone
from sklearn.model_selection import KFold


class Models(object):
    """
    A class abstracting away a bunch of models. predict_proba and predict_proba
    returns all the individual predictions stacked on a new dimension.
    """

    def __init__(self, models):
        self._models = models

    def fit(self, X, y):
        """
        Fits all the underlying models
        :param X: A (n_samples,  n_features) array
        :param y: The input target (n_samples,)
        :return: a fitted version of self
        """
        for model in self._models:
            model.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Returns an array of predictions probabilities. All the predictions of
        the underlying models are stacked on a new dimension.
        :param X: A (n_samples,  n_features) array
        :return: A (n_samples, n_classes, n_models) dimensions array
        """
        return np.stack(
            [model.predict_proba(X) for model in self._models],
            axis=-1
        )

    def predict(self, X):
        """
        Returns an array of predictions. All the predictions of the underlying
        models are stacked on a new dimension
        :param X: A (n_samples,  n_features) array
        :return: A (n_samples, n_models) dimensions array
        """
        return np.stack([model.predict_proba(X) for model in self._models],
                        axis=-1)

    def __getitem__(self, item):
        """
        Models indexing method
        :param item: int
        :return: model, as passed to constructor
        """
        return self._models[item]


class CrossValidatedModel(object):
    """
    A class abstracting away the cross-validation step. The base model given
    as input is cloned n_splits times so that each clone will be trained on a
    subset of the input data while calling the fit method. The each prediction
    is averaged when calling the predict_proba and predict_proba methods.
    """
    def __init__(self, model, n_splits=3):
        self.base_model = model
        self._models = [clone(model) for _ in range(n_splits)]
        self.n_splits = n_splits

    def fit(self, X, y):
        """
        Fits all the underlying models
        :param X: A (n_samples,  n_features) array
        :param y: The input target (n_samples,)
        :return: a fitted version of self
        """
        for model in self._models:
            model.fit(X, y)
        return self

    def predict_proba(self, X):
        """
        Returns an array of predictions probabilities. All the predictions of
        the underlying models are stacked on a new dimension.
        :param X: A (n_samples,  n_features) array
        :return: A (n_samples, n_classes, n_splits) dimensions array
        """
        predictions = np.full(
            (len(X), len(self._models[0].classes_), self.n_splits),
            np.nan
        )
        kfold = KFold(n_splits=self.n_splits)
        for i, (train_index, _) in enumerate(kfold.split(X)):
            X_train = X[train_index]
            model = self._models[i]
            prediction = model.predict_proba(X_train)
            predictions[train_index, :, i] = prediction
        mean_prediction = np.nanmean(predictions, axis=-1)
        return mean_prediction

    def predict(self, X):
        """
        Returns an array of predictions. All the predictions of the underlying
        models are stacked on a new dimension
        :param X: A (n_samples,  n_features) array
        :return: A (n_samples, n_splits) dimensions array
        """
        prediction_proba = self.predict_proba(X)
        return np.argmax(prediction_proba, axis=-1)
