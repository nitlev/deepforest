from mock import MagicMock
from sklearn.metrics import roc_auc_score

from deepforest.cascadeforest import CascadeForest
from tests.utils import TestWithData, models_generator


class TestgcForest(TestWithData):
    def test_gcforest_grow_should_add_level_to_gcforest(self):
        # Given
        gcforest = CascadeForest(models_generator(self.y_train), roc_auc_score)

        # When
        gcforest.grow(self.X_train, self.y_train, self.X_test, self.y_test)

        # Check
        assert gcforest.levels > 0

    def test_gcforest_last_layer_should_be_trained_after_grow(self):
        # Given
        gcforest = CascadeForest(models_generator(self.y_train), roc_auc_score)

        # When
        gcforest.grow(self.X_train, self.y_train, self.X_test, self.y_test)

        # Check
        gcforest.output_layer.models[0].fit.assert_called()

    def test_predict_proba_should_return_properly_formatted_array(self):
        # Given
        gcforest = CascadeForest(models_generator(self.y_train), roc_auc_score)
        gcforest.grow(self.X_train, self.y_train, self.X_test, self.y_test)

        # When
        predictions = gcforest.predict_proba(self.X_test)

        # Check
        assert predictions.shape == (len(self.X_test), 2)

    def test_predict_should_return_properly_formated_array(self):
        # Given
        gcforest = CascadeForest(models_generator(self.y_train), roc_auc_score)
        gcforest.grow(self.X_train, self.y_train, self.X_test, self.y_test)

        # When
        predictions = gcforest.predict(self.X_test)

        # Check
        assert predictions.shape == (len(self.X_test),)

    def test_fit_should_call_output_layer_fit(self):
        # Given
        gcforest = CascadeForest(models_generator(self.y_train), roc_auc_score)
        gcforest.grow(self.X_train, self.y_train, self.X_test, self.y_test)
        gcforest.output_layer = MagicMock()

        # When
        gcforest.fit(self.X_test, self.y_test)

        # Check
        gcforest.output_layer.fit.assert_called_once_with(self.X_test,
                                                          self.y_test)
