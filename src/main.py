from input import sample_get_input
from model import keras
from util import graph_functions

# First get our data and preprocess it.
(train, test) = sample_get_input.get_input()

# Then train our model.
nn = keras.build_network()
# Needed for producing train/test loss/accuracy graphs.
history = keras.train_model(nn, train, test)

# Finally create our output (graphs/predictions).
graph_functions.plot_loss(history.history, "loss_plot")
graph_functions.plot_accuracy(history.history, "accuracy_plot")



print(":)")
