import keras
from keras.models import Sequential
from keras.layers import Dense

# Hyperparams
batch_size = 10
num_classes = 2
epochs = 100
learning_rate = 0.001

def build_network():
    '''Builds a Keras MLP and returns the compiled model.'''
    model = Sequential()

    model.add(Dense(5, activation='relu', input_shape=(2,)))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
            optimizer='sgd',
            metrics=['accuracy'])
    return model


def train_model(model, train, test):
    '''Trains a compiled model on X and y and validates on X_test and y_test.
    Returns the model's history.'''
    X = train.ix[:,:-1].as_matrix()
    y = keras.utils.to_categorical(train.ix[:,-1], num_classes)
    X_test= test.ix[:,:-1].as_matrix()
    y_test = keras.utils.to_categorical(test.ix[:,-1], num_classes)

    return model.fit(X, y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_test, y_test))

def evaluate_model(model, X_test, y_test):
    '''Returns a trained models performance on test data.'''
    return model.evaluate(X_test, y_test, verbose=0)
