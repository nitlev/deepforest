import numpy as np
from mock import MagicMock


def prepare_x(shape):
    return np.random.rand(*shape)


def prepare_y(shape):
    return np.random.randint(0, 2, shape)


def load_data():
    return prepare_x((10, 10)), prepare_y(10), prepare_x((5, 5)), prepare_y(5)


def create_models(n, predicted_value):
    models = []
    for i in range(n):
        new_model = MagicMock()
        new_model.predict.return_value = predicted_value
        new_model.predict_proba.return_value = np.stack([predicted_value,
                                                         1 - predicted_value],
                                                        axis=1)
        models.append(new_model)
    return models
