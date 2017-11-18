
class BaseModel():
    def __init__(self):
        raise NotImplementedError

    def construct_model(self):
        raise NotImplementedError

    def train(self, x_train, y_train, x_test, y_test):
        raise NotImplementedError

    def test(self, x_test, y_test):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError
