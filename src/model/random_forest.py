from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from .model import BaseModel

class RandomForest(BaseModel):
    def __init__(self, n_estimators):
        BaseModel.__init__(self)
        self._n_estimators = n_estimators
        self.model = self.construct_model()

    def construct_model(self):
        return RandomForestRegressor(n_estimators=self._n_estimators)

    def train(self, x_train, y_train, x_test, y_test):
        self.model.fit(x_train, y_train)

    def test(self, x_test, y_test):
        score = cross_val_score(self.model, x_test, y_test)
        print("Score: " % score)

    def predict(self, x):
        return self.model.predict(x)

