import numpy as np
import pandas as pd


def check_models(models):
    at_least_one_model = False
    for model in models:
        check_model(model)
        at_least_one_model = True
    assert at_least_one_model, "You must pass at least one model to a Layer"
    return models


def check_model(model):
    has_fit_method = hasattr(model, "fit")
    has_predict_method = hasattr(model, "predict")
    assert has_fit_method and has_predict_method, \
        "models should have fit and predict methods"


def predictions_to_dataframe(predictions, index, n_models):
    n_class = predictions[0].shape[1]
    columns = ["model{}_{}".format(model_number, class_number) for model_number
               in range(n_models) for class_number in range(n_class)]
    concat_dataframe = np.concatenate(predictions, axis=1)
    return pd.DataFrame(concat_dataframe, columns=columns, index=index)
