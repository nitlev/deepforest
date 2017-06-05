import numpy as np

from deepforest.scan import Scan, MultiGrainedScan
from .utils import TestWithData, create_models, prepare_x, prepare_y


class Test(TestWithData):
    def setup(self):
        self.X_train = prepare_x((100, 3, 3))
        self.y_train = prepare_y((100,))

    def test_fit_transform_should_return_2_dim_array(self):
        # Given
        n_models = 2
        n_patches = 3
        models = create_models(n=n_models, predicted_value=self.y_train)
        scanning = Scan(models, n_patches, patch_size=(2, 2))

        # When
        result = scanning.fit_transform(self.X_train, self.y_train)

        # Check
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 2

    def test_fit_transform_should_return_array_with_proper_dimensions(self):
        # Given
        n_classes = 2
        n_models = 2
        n_patches = 3
        models = create_models(n=n_models,
                               predicted_value=np.repeat(self.y_train,
                                                         n_patches, axis=0))
        scanning = Scan(models, n_patches, patch_size=(2, 2))

        # When
        result = scanning.fit_transform(self.X_train, self.y_train)

        # Check
        assert result.shape[1] == n_classes * n_patches * n_models

    def test_fit_transform_should_return_properly_formatted_array_when_multiclass(self):
        # Given
        n_classes = 3
        n_models = 2
        n_patches = 3
        y_train_multiclass = prepare_y((100, n_classes))
        predicted_value = np.repeat(y_train_multiclass, n_patches, axis=0)
        models = create_models(n=n_models,
                               predicted_value=predicted_value, n_class=3)
        scanning = Scan(models, n_patches, patch_size=(2, 2))

        # When
        result = scanning.fit_transform(self.X_train, y_train_multiclass)

        # Check
        assert result.shape[1] == n_classes * n_patches * n_models

    def test_scan_should_return_properly_formatted_array(self):
        # Given
        n_models = 2
        n_patches = 3
        patch_size = 2
        models = create_models(n=n_models,
                               predicted_value=self.y_train)
        scanning = Scan(models, n_patches,
                        patch_size=(patch_size, patch_size))

        # When
        result = scanning._scan(self.X_train)

        # Check
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 2
        assert result.shape == (len(self.X_train) * n_patches,
                                patch_size * patch_size)

    def test_scan_with_y_parameter_should_return_two_properly_sized_arrays(
            self):
        # Given
        n_models = 2
        n_patches = 3
        patch_size = 2
        models = create_models(n=n_models, predicted_value=self.y_train)
        scanning = Scan(models, n_patches,
                        patch_size=(patch_size, patch_size))

        # When
        result_x, result_y = scanning._scan(self.X_train, self.y_train)

        # Check
        assert isinstance(result_x, np.ndarray)
        assert len(result_x.shape) == 2
        assert result_x.shape == (len(self.X_train) * n_patches,
                                  patch_size * patch_size)

        # And
        assert isinstance(result_y, np.ndarray)
        assert len(result_y.shape) == len(self.y_train.shape)
        assert result_y.shape[0] == len(self.X_train) * n_patches

    def test_fit_should_call_underlying_fit_method(self):
        # Given
        n_models = 2
        n_patches = 3
        patch_size = 2
        models = create_models(n=n_models, predicted_value=self.y_train)
        scanning = Scan(models, n_patches,
                        patch_size=(patch_size, patch_size))

        # When
        scanning.fit(self.X_train, self.y_train)

        # Check
        for model in models:
            model.fit.assert_called_once()

    def test_transform_should_call_underlying_predict_proba_method(self):
        # Given
        n_models = 2
        n_patches = 3
        patch_size = 2
        models = create_models(n=n_models, predicted_value=self.y_train)
        scanning = Scan(models, n_patches,
                        patch_size=(patch_size, patch_size))

        # When
        scanning.transform(self.X_train)

        # Check
        for model in models:
            model.predict_proba.assert_called_once()


class TestMultiGrainedScanning(TestWithData):
    def setup(self):
        self.X_train = prepare_x((100, 3, 3))
        self.y_train = prepare_y((100,))

    def test_fit_transform_should_return_properly_formatted_array(self):
        # Given
        n_classes = 2
        n_models = 2
        n_patches = 3
        patch_size = 2
        n_grains = 3
        predicted_value = np.repeat(self.y_train, n_patches, axis=0)
        models = create_models(n=n_models, predicted_value=predicted_value)
        mgs = MultiGrainedScan(models=models,
                               n_patch=n_patches,
                               patch_sizes=((patch_size, patch_size),) * n_grains)

        # When
        result = mgs.fit_transform(self.X_train, self.y_train)

        # Check
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 2
        assert result.shape[1] == n_classes * n_patches * n_models * n_grains

    def test_transform_should_call_models_predict_proba(self):
        # Given
        n_models = 2
        n_patches = 3
        patch_size = 2
        n_grains = 3
        predicted_value = np.repeat(self.y_train, n_patches, axis=0)
        models = create_models(n=n_models, predicted_value=predicted_value)
        mgs = MultiGrainedScan(models=models,
                               n_patch=n_patches,
                               patch_sizes=((patch_size,
                                             patch_size),) * n_grains)

        # When
        _ = mgs.transform(self.X_train)

        # Check
        for scan in mgs.scans:
            for model in scan.models:
                model.predict_proba.assert_called_once()

    def test_fit_should_call_models_fit(self):
        # Given
        n_models = 2
        n_patches = 3
        patch_size = 2
        n_grains = 3
        predicted_value = np.repeat(self.y_train, n_patches, axis=0)
        models = create_models(n=n_models, predicted_value=predicted_value)
        mgs = MultiGrainedScan(models=models,
                               n_patch=n_patches,
                               patch_sizes=((patch_size,
                                             patch_size),) * n_grains)

        # When
        _ = mgs.fit(self.X_train, self.y_train)

        # Check
        for scan in mgs.scans:
            for model in scan.models:
                model.fit.assert_called_once()
