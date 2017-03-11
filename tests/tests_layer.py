from __future__ import absolute_import

import numpy as np
import pytest
from mock import MagicMock

from deepforest.layer import Layer, InputLayer
from .utils import load_data, create_models


class TestLayer(object):
    def setup(self):
        self.X_train, self.y_train, \
        self.X_test, self.y_test = load_data()

    def test_layer_can_be_fitted_on_dataframe(self):
        # Given
        model = MagicMock()
        layer = InputLayer(model)

        # When
        new_layer = layer.fit(self.X_train, self.y_train)

        # Check
        assert isinstance(new_layer, InputLayer)

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

    def test_input_layer_predict_should_return_properly_formated_array(self):
        # Given
        models = create_models(n=3, predicted_value=self.y_test)
        layer = InputLayer(*models)

        # When
        predict = layer.predict(self.X_test)

        # Check
        assert predict.shape == (len(self.X_test), 3)

    def test_layer_predict_should_return_properly_formated_array(self):
        # Given
        models = create_models(n=3, predicted_value=self.y_test)
        input_layer = create_models(n=1, predicted_value=self.y_test)[0]
        layer = Layer(input_layer, *models)

        # When
        predict = layer.predict(self.X_test)

        # Check
        assert predict.shape == (len(self.X_test), 6)

    def test_layer_fit_should_call_previous_layers_fit_method(self):
        # Given
        hidden_models = create_models(3, self.y_train)
        input_layer = MagicMock()
        input_layer.predict.return_value = self.y_train
        hidden_layer = Layer(input_layer, *hidden_models)

        # When
        hidden_layer.fit(self.X_train, self.y_train)

        # Check
        input_layer.fit.assert_called_once_with(self.X_train, self.y_train)

    def test_layer_fit_should_call_previous_layer_predict_method(self):
        # Given
        hidden_models = create_models(3, predicted_value=self.y_train)
        input_layer = MagicMock()
        input_layer.predict.return_value = self.y_train
        hidden_layer = Layer(input_layer, *hidden_models)

        # When
        hidden_layer.fit(self.X_train, self.y_train)

        # Check
        input_layer.predict.assert_called_once_with(self.X_train)

    def test_layer_predict_should_call_predict_on_augmented_dataset(self):
        # Given
        hidden_models = create_models(3, predicted_value=self.y_train)
        input_layer = MagicMock()
        predictions = np.stack([self.y_train for _ in range(3)], axis=-1)
        input_layer.predict.return_value = predictions
        hidden_layer = Layer(input_layer, *hidden_models)

        # When
        hidden_layer.fit(self.X_train, self.y_train)

        # Check
        for model in hidden_models:
            args, kwargs = model.fit.call_args
            assert args[0].shape == (self.X_train.shape[0],
                                     self.X_train.shape[1] + 3)
