try:
    from unittest.mock import MagicMock
except ImportError:
    from mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from deepforest.layer import Layer, InputLayer


def split_x_y(dataframe, target):
    return dataframe.drop(target, axis=1), dataframe[target]


def prepare_x(shape):
    data = np.random.rand(*shape)
    return pd.DataFrame(data)


def prepare_y(shape):
    data = np.random.randint(0, 2, shape)
    return pd.Series(data)


def load_data():
    return prepare_x((10, 10)), prepare_y(10), prepare_x((5, 5)), prepare_y(5)


class TestLayer(object):
    def setup(self):
        self.X_train, self.y_train, \
        self.X_test, self.y_test = load_data()

    def create_models(self, n, predicted_value):
        models = []
        for i in range(n):
            new_model = MagicMock()
            new_model.predict_proba.return_value = pd.concat([predicted_value,
                                                              1 - predicted_value],
                                                             axis=1)
            models.append(new_model)
        return models

    def test_layer_can_be_fitted_on_dataframe(self):
        # Given
        model = MagicMock()
        layer = InputLayer(model)

        # When
        layer.fit(self.X_train, self.y_train)

        # Check
        assert isinstance(layer, Layer)

    def test_layer_should_throw_exception_error_if_no_model_is_given(self):
        # Check
        with pytest.raises(AssertionError):
            # When
            layer = Layer(layer=None)

    def test_layer_should_throw_exception_if_bad_model_is_given(self):
        # Check
        with pytest.raises(AssertionError):
            # When
            layer = InputLayer(None)

    def test_layer_fit_should_fit_underlying_models(self):
        # Given
        model = MagicMock()
        layer = InputLayer(model)

        # When
        layer.fit(self.X_train, self.y_train)

        # Check
        model.fit.assert_called_once_with(self.X_train, self.y_train)

    def test_layer_predict_should_return_properly_formated_array(self):
        # Given
        models = self.create_models(n=3, predicted_value=self.y_test)
        layer = InputLayer(*models)

        # When
        predict = layer.predict(self.X_test)

        # Check
        assert predict.shape == (len(self.X_test), 6)

    def test_layer_fit_should_call_previous_layers_fit_method(self):
        # Given
        hidden_models = self.create_models(3, self.y_train)
        input_layer = MagicMock()
        input_layer.predict.return_value = self.y_train
        hidden_layer = Layer(input_layer, *hidden_models)

        # When
        hidden_layer.fit(self.X_train, self.y_train)

        # Check
        input_layer.fit.assert_called_once_with(self.X_train, self.y_train)

    def test_layer_fit_should_call_previous_layer_predict_method(self):
        # Given
        hidden_models = self.create_models(3, predicted_value=self.y_train)
        input_layer = MagicMock()
        input_layer.predict.return_value = self.y_train
        hidden_layer = Layer(input_layer, *hidden_models)

        # When
        hidden_layer.fit(self.X_train, self.y_train)

        # Check
        input_layer.predict.assert_called_once_with(self.X_train)
