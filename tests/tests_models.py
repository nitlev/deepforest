from __future__ import absolute_import

from mock import MagicMock

from deepforest.models import Models, CrossValidatedModel
from .utils import create_models, TestWithData


class TestModels(TestWithData):
    def test_Models_fit_method_should_fit_all_internal_models(self):
        # Given
        models = Models(create_models(n=3, predicted_value=self.y_test))

        # When
        models.fit(self.X_train, self.y_train)

        # Check
        for model in models._models:
            model.fit.assert_called_once_with(self.X_train, self.y_train)

    def test_Models_fit_method_should_return_self(self):
        # Given
        models = Models(create_models(n=3, predicted_value=self.y_test))

        # When
        new_models = models.fit(self.X_train, self.y_train)

        # Check
        assert models is new_models

    def test_Models_predict_method_should_call_predict_on_all_models(self):
        # Given
        models = Models(create_models(n=3, predicted_value=self.y_test))

        # When
        models.predict(self.X_test)

        # Check
        for model in models._models:
            model.predict.assert_called_once_with(self.X_test)

    def test_Models_predict_should_return_properly_formatted_array(self):
        # Given
        nb_models = 3
        models = Models(create_models(n=nb_models,
                                      predicted_value=self.y_test))

        # When
        prediction = models.predict(self.X_test)

        # Check
        assert prediction.shape == (len(self.y_test), nb_models)

    def test_Models_predict_proba_method_should_call_predict_proba_on_all_models(
            self):
        # Given
        models = Models(create_models(n=3,
                                      predicted_value=self.y_test))

        # When
        models.predict_proba(self.X_test)

        # Check
        for model in models._models:
            model.predict_proba.assert_called_once_with(self.X_test)

    def test_Models_predict_proba_should_return_properly_formatted_array(self):
        # Given
        nb_models = 3
        models = Models(create_models(n=nb_models,
                                      predicted_value=self.y_test))

        # When
        prediction = models.predict_proba(self.X_test)

        # Check
        assert prediction.shape == (len(self.y_test), 2 * nb_models)


class TestCrossValidatedModel(TestWithData):
    def test_cvmodel_fit_should_fit_underlying_mode(self):
        # Given
        model_mock = MagicMock()
        cvmodel = CrossValidatedModel(model_mock)

        # When
        cvmodel.fit(self.X_train, self.y_train)

        # Check
        for model in cvmodel._models:
            model.fit.assert_called()

    def test_cvmodel_fit_should_return_crossvalmodel(self):
        # Given
        model_mock = MagicMock()
        cvmodel = CrossValidatedModel(model_mock)

        # When
        new_model = cvmodel.fit(self.X_train, self.y_train)

        # Check
        assert isinstance(new_model, CrossValidatedModel)

    def test_csvmdoel_predict_proba_should_return_properly_formatted_array(self):
        # Given
        n_splits = 3
        model_mock = MagicMock()
        cvmodel = CrossValidatedModel(model_mock, n_splits=n_splits)
        cvmodel._models = create_models(n_splits, self.y_test)

        # When
        prediction = cvmodel.predict_proba(self.X_test)

        # Check
        assert prediction.shape == (len(self.y_test), 2)

    def test_csvmdoel_predict_should_return_properly_formatted_array(self):
        # Given
        n_splits = 3
        model_mock = MagicMock()
        cvmodel = CrossValidatedModel(model_mock, n_splits=n_splits)
        cvmodel._models = create_models(n_splits, self.y_test)

        # When
        prediction = cvmodel.predict(self.X_test)

        # Check
        assert prediction.shape == (len(self.y_test),)
