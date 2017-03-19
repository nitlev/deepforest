import numpy as np
from mock import MagicMock


def prepare_x(shape):
    return np.random.rand(*shape)


def prepare_y(shape):
    return np.random.randint(0, 2, shape)


def load_data():
    return prepare_x((10, 10)), prepare_y(10), prepare_x((5, 5)), prepare_y(5)


def create_models(n, predicted_value):
    def predict(X):
        return predicted_value[:len(X)]

    def predict_proba(X):
        return np.stack([predict(X),
                         1 - predict(X)],
                        axis=1)

    models = []
    for i in range(n):
        new_model = MagicMock()
        new_model.classes_ = [0, 1]
        new_model.predict.side_effect = predict
        new_model.predict_proba.side_effect = predict_proba
        models.append(new_model)
    return models
