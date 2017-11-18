import keras
from keras.models import Sequential
from keras.layers import Dense

from .model import BaseModel

class Kegression(BaseModel):
    def __init__(self, batch_size, epochs, learning_rate = 0.001):
        BaseModel.__init__(self)

        # Hyperparameters
        self._batch_size = batch_size
        self._epochs = epochs
        self._learning_rate = learning_rate

        self.model = self.construct_model()

    def construct_model(self):
        '''Builds a Keras MLP and returns the compiled model.'''
        model = Sequential()

        model.add(Dense(5, activation='relu', input_shape=(8,)))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(5, activation='relu'))
        model.add(Dense(1))

        model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['mse'])
        return model

    def train(self, x_train, y_train, x_test, y_test):
        '''Trains a compiled model on X and y and validates on X_test and y_test.
        Returns the model's history.'''

        # TODO: Move into a static file for preparing data
        # X = train.ix[:, :-1].as_matrix()
        # y = keras.utils.to_categorical(train.ix[:, -1], self.num_classes)
        # X_test = test.ix[:, :-1].as_matrix()
        # y_test = keras.utils.to_categorical(test.ix[:, -1], self.num_classes)

        return self.model.fit(x_train, y_train,
                         batch_size=self._batch_size,
                         epochs=self._epochs,
                         verbose=1)

    def test(self, x_test, y_test):
        loss = self.model.evaluate(x_test, y_test, verbose=0)

        print("Loss: %.2f" % loss)

    def predict(self, x):
        return self.model.predict(x, batch_size=self._batch_size)
