'''File to store all the shared functionality for graphing and pickling.'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns

# Plotting functions. All use SVG as they are small and can zoom.
def plot_mean_squared_error(hist_obj, filename, title="Mean Squared Error"):
    '''Plots the train/test accuracy over epochs for hist_obj.'''
    plt.plot(hist_obj['mean_squared_error'])
    plt.plot(hist_obj['val_mean_squared_error'])
    plt.title(title)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("graph/{}.svg".format(filename), format='svg')
    plt.clf()


def plot_loss(hist_obj, filename, title="Loss"):
    '''Plots the train/test loss over epochs for hist_obj.'''
    plt.plot(hist_obj['loss'])
    plt.plot(hist_obj['val_loss'])
    plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("graph/{}.svg".format(filename), format='svg')
    plt.clf()


def dump_history(hist_obj, filename):
    '''Dumps a history object to a pickle.'''
    with open("dumps/{}.pickle".format(filename), 'wb') as fd:
        pickle.dump(hist_obj, fd)

def plot_with_seaborn(hist_obj):
    df = pd.DataFrame.from_dict(hist_obj)

    '''Makes nicer plots using Seaborn.'''
    # Set plot styles using Seaborn.
    sns.set_style("dark")
    sns.set_context("poster")
    # Set the colours. As there are only 3 colours it will loop around.
    sns.set_palette("muted", n_colors=1)

    # Create subplots.
    f, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # Alpha channel percentage for train plots.
    train_alpha = 0.5
    # Line weights for the plots.
    train_lw = 4
    test_lw = 5

    # Plot the loss.
    axes[0].plot(df['val_loss'], lw=test_lw)
    axes[0].plot(df['loss'], alpha=train_alpha, lw=train_lw)
    axes[0].set_title("Loss")

    # Plot the accuracy.
    axes[1].plot(df['val_mean_squared_error'], lw=test_lw)
    axes[1].plot(df['mean_squared_error'], alpha=train_alpha, lw=train_lw)
    axes[1].set_title("Accuracy")

    # Set the legend.
    axes[0].legend(['Model 1'])

    # Remove the axis lines.
    sns.despine(left=True, bottom=True)

    # Remove the x labels.
    plt.setp(axes, xticks=[])
    plt.tight_layout()

    # Save the output figure.
    plt.savefig("graph/combined_graphs.png")
