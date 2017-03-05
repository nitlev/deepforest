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


class Layer(object):
    def __init__(self, models):
        self.models = check_models(models)

    def fit(self, X, y):
        return self
