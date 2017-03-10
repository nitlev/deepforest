import numpy as np
import pandas as pd
import pytest

from deepforest.utils import predictions_to_dataframe, check_model


class TestUtils(object):
    def test_predictions_to_dataframe_should_convert_list_of_array_to_dataframe(
            self):
        # Given
        n_models = 3
        arrays = [np.random.rand(2, 4) for i in range(n_models)]

        # When
        dataframe = predictions_to_dataframe(arrays, (0, 1), n_models)

        # Check
        assert isinstance(dataframe, pd.DataFrame)
        assert dataframe.shape == (2, 12)

    def test_check_model_should_raise_error_if_object_has_no_fit_method(self):
        # Given
        class StubPredict:
            def predict(self):
                pass

        stubPredict = StubPredict()

        # Check
        with pytest.raises(AssertionError):
            # When
            check_model(stubPredict)

    def test_check_model_should_raise_error_if_object_has_no_predict_method(self):
        # Given
        class StubFit:
            def fit(self):
                pass

        stubFit = StubFit()

        # Check
        with pytest.raises(AssertionError):
            # When
            check_model(stubFit)
