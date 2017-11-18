from input import input
from model.keras import Kegression
from util.graph import graph_functions
from util import constants as const

from model import get_model



def run():
    #############################
    ###         Data          ###
    #############################
    (x_train, y_train, x_test, y_test, id) = input.get_input()

    ##############################
    ###    Keras Regression    ###
    ##############################

    keg = get_model.from_name("kegression")
    kegression = keg(const.BATCH_SIZE, const.EPOCHS, const.LEARNING_RATE)
    # Needed for producing train/test loss/accuracy graphs.
    # KERAS ONLY
    history = kegression.train(x_train, y_train, x_test, y_test)

    ################################
    ### Random Forest Regression ###
    ################################

    # rf = get_model.from_name("random_forest")
    # random_forest = rf(const.N_ESTIMATORS)
    # random_forest.train(x_train, y_train, x_test, y_test)

    #############################
    ###    Output & Graphs    ###
    #############################
    # Finally create our output (graphs/predictions).
    graph_functions.plot_loss(history.history, "loss_plot")
    graph_functions.plot_mean_squared_error(history.history, "mse_plot")

#    graph_functions.plot_with_seaborn(history.history)

    print(":)")

if __name__ == "__main__":
    run()

