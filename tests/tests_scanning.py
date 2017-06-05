import numpy as np

from deepforest.scanning import Scanning
from .utils import TestWithData


class Test(TestWithData):
    def test_scanning_fit_transform_should_return_2_dim_array(self):
        # Given
        scanning = Scanning()

        # When
        result = scanning.fit_transform(self.X_train, self.y_train)

        # Check
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 2
