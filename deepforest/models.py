import numpy as np


class Models(object):
    def __init__(self, models):
        self._models = models

    def fit(self, X, y):
        for model in self._models:
            model.fit(X, y)
        return self

    def predict(self, X):
        return np.stack([model.predict(X) for model in self._models],
                        axis=-1)

    def predict_proba(self, X):
        return np.concatenate(
            [model.predict_proba(X) for model in self._models],
            axis=1
        )
