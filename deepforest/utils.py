def check_models(models):
    at_least_one_model = False
    for model in models:
        check_model(model)
        at_least_one_model = True
    assert at_least_one_model, "You must pass at least one model to a Layer"
    return models


def check_model(model):
    assert_has_method(model, "fit")
    assert_has_method(model, "predict_proba")
    assert_has_method(model, "predict_proba")


def assert_has_method(model, method):
    assert hasattr(model, method), \
        "models should have {} method, type {} doesn't".format(method,
                                                               type(model))
