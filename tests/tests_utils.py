import numpy as np
import pandas as pd

from deepforest.utils import predictions_to_dataframe


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
