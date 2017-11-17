'''File to store all the shared functionality for graphing and pickling.'''

import matplotlib.pyplot as plt
import numpy as np
import pickle


# Plotting functions. All use SVG as they are small and can zoom.
def plot_accuracy(hist_obj, filename, title="Accuracy"):
    '''Plots the train/test accuracy over epochs for hist_obj.'''
    plt.plot(hist_obj['acc'])
    plt.plot(hist_obj['val_acc'])
    plt.title(title)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("../graph/{}.svg".format(filename), format='svg')
    plt.clf()


def plot_loss(hist_obj, filename, title="Loss"):
    '''Plots the train/test loss over epochs for hist_obj.'''
    plt.plot(hist_obj['loss'])
    plt.plot(hist_obj['val_loss'])
    plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("../graph/{}.svg".format(filename), format='svg')
    plt.clf()


def dump_history(hist_obj, filename):
    '''Dumps a history object to a pickle.'''
    with open("dumps/{}.pickle".format(filename), 'wb') as fd:
        pickle.dump(hist_obj, fd)
