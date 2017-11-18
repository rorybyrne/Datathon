from input import input
from model.keras import Kegression
from util.graph import graph_functions
from util import constants as const

from model import get_model

from sklearn.metrics import mean_squared_error

import numpy as np

import pandas as pd



def run():
    #############################
    ###         Data          ###
    #############################
    (x_train, y_train, x_test, y_test, id) = input.get_input()

    ##############################
    ###    Keras Regression    ###
    ##############################

    # keg = get_model.from_name("kegression")
    # kegression = keg(const.BATCH_SIZE, const.EPOCHS, const.LEARNING_RATE)
    #
    # # Needed for producing train/test loss/accuracy graphs.
    # # KERAS ONLY
    # np.savetxt("out/x_train.csv", x_train, delimiter="\t")
    # history = kegression.train(x_train, y_train, x_test, y_test)
    #
    # preds = kegression.predict(x_test)
    #
    # out_df = pd.DataFrame()
    # for i, j, k in zip(preds, y_test, id):
    #     s = pd.Series(data=[i[0], j[0], k])
    #     out_df = out_df.append(s, ignore_index=True)
    #     print("Guess %s || Actual %s" % (i[0], j[0]))
    #
    # out_df.to_csv("out/out.csv", header=None, index=False)

    ################################
    ### Random Forest Regression ###
    ################################

    rf = get_model.from_name("random_forest")
    random_forest = rf(const.N_ESTIMATORS)
    random_forest.train(x_train, y_train, x_test, y_test)

    total_guess = []
    total_actual = []
    for i in range(len(x_test)):
        guess = random_forest.predict([x_test[i, :]])
        actual = y_test[i, :]

        total_guess.append(guess)
        total_actual.append(actual)

    print(mean_squared_error(total_actual, total_guess))

    #############################
    ###    Output & Graphs    ###
    #############################
    # Finally create our output (graphs/predictions).
    # graph_functions.plot_loss(history.history, "loss_plot")
    # graph_functions.plot_mean_squared_error(history.history, "mse_plot")

#    graph_functions.plot_with_seaborn(history.history)

    print(":)")

if __name__ == "__main__":
    run()

