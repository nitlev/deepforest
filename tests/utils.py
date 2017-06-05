import numpy as np
from mock import MagicMock


class TestWithData(object):
    def setup(self):
        self.X_train, self.y_train, \
        self.X_test, self.y_test = load_data()


def load_data():
    return prepare_x((100, 100)), prepare_y(100),\
           prepare_x((50, 50)), prepare_y(50)


def prepare_x(shape):
    return np.random.rand(*shape)


def prepare_y(shape):
    return np.random.randint(0, 2, shape)


def models_generator(predicted_value):
    while True:
        yield create_models(3, predicted_value)


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
        new_model.fit.return_value = new_model
        new_model.classes_ = [0, 1]
        new_model.predict.side_effect = predict
        new_model.predict_proba.side_effect = predict_proba
        models.append(new_model)
    return models
