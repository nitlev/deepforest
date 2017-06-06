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

    def test_has_improved_should_return_true_iif_score_went_up_when_maximizing(self):
        # Given
        gcforest = CascadeForest(models_generator(self.y_train),
                                 roc_auc_score,
                                 objective='maximize')

        # When
        has_improved_true = gcforest._has_improved(2, 1)
        has_improved_false = gcforest._has_improved(1, 2)
        has_improved_false2 = gcforest._has_improved(1, 1)

        # Check
        assert has_improved_true
        assert not has_improved_false
        assert not has_improved_false2

    def test_has_improved_should_return_true_iif_score_went_up_when_minimizing(self):
        # Given
        gcforest = CascadeForest(models_generator(self.y_train),
                                 roc_auc_score,
                                 objective='minimize')

        # When
        has_improved_true = gcforest._has_improved(1, 2)
        has_improved_false = gcforest._has_improved(2, 1)
        has_improved_false2 = gcforest._has_improved(1, 1)

        # Check
        assert has_improved_true
        assert not has_improved_false
        assert not has_improved_false2
