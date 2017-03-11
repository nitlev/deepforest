import pytest

from deepforest.utils import check_model


class TestUtils(object):
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

    def test_check_model_should_raise_error_if_object_has_no_predict_method(
            self):
        # Given
        class StubFit:
            def fit(self):
                pass

        stubFit = StubFit()

        # Check
        with pytest.raises(AssertionError):
            # When
            check_model(stubFit)
