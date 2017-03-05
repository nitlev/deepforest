import pandas as pd
from sklearn.model_selection import train_test_split

from deepforest.layer import Layer


def split_x_y(dataframe, target):
    return dataframe.drop(target, axis=1), dataframe[target]


def prepare_data(dataframe: pd.DataFrame):
    clean_data = dataframe.drop(["Cabin", "Name", "PassengerId", "Ticket"],
                                axis=1)
    clean_data = pd.get_dummies(clean_data).fillna(-1)
    train, test = train_test_split(clean_data)
    X_train, y_train = split_x_y(train, "Survived")
    X_test, y_test = split_x_y(test, "Survived")
    return X_train, y_train, X_test, y_test


def load_data(path: str):
    raw_dataframe = pd.read_csv(path)
    return prepare_data(raw_dataframe)


class TestLayer(object):
    def setup(self):
        self.X_train, self.y_train, \
        self.X_test, self.y_test = load_data("data/train.csv")

    def test_layer_can_be_fitted_on_dataframe(self):
        # Given
        layer = Layer()

        # When
        layer.fit(self.X_train, self.y_train)

        # Check
        assert isinstance(layer, Layer)
