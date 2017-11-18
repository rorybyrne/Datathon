import pandas as pd
from util.constants import TOTAL_TRAINING
from features import preprocess as pp

from pprint import pprint


def train_test_split(df, train_percent=0.8):
    '''Randomly samples and splits data into train/test split.'''
    # Uncomment these lines to maintain ordering:
    #split_index = int(df.shape[0] * train_percent)
    #return (df[:split_index], df[split_index:])

    train = df.sample(frac=0.8, random_state=200)
    test = df.drop(train.index)

    return (train, test)

def get_input():
    '''
    1) Get raw data
    2) Drop NaNs from raw
    3) Get ID col
    4) Select all relevant cols
    5) Select cols to normalise
    6) Select cols we can't normalise
    7) Normalise to_normalise
    8) Concat with not_normalise
    9) Test/train split
    :return:
    '''
    total_raw = load_csv(TOTAL_TRAINING)

    # Drop NaNs
    total_dropped_nan = pp.drop_nan(total_raw)

    # Get ID col
    ids = total_dropped_nan['ID']

    # Get relevant cols
    selected = total_dropped_nan[['Weekday', 'Month', 'AWND', 'PRCP', 'SNOW', 'TAVG', 'TMAX', 'TMIN', 'BikeRides']]

    # Select cols to normalise
    to_normalise = selected[['AWND', 'PRCP', 'SNOW', 'TAVG', 'TMAX', 'TMIN']]
    print("to_normalise shape: (%s, %s)" % to_normalise.shape)

    # Select non-normalise cols
    not_normalise = selected[['Weekday', 'Month', 'BikeRides']]
    print("not_normalise shape: (%s, %s)" % not_normalise.shape)

    # Normalise cols
    normalised = pp.normalise(to_normalise)

    # Concat
    total_normalised = not_normalise.append(normalised, axis=1)
    print("Total normalised shape: (%s, %s)" % total_normalised.shape)
    pprint(total_normalised)

    # pprint(total_normalised)

    train, test = train_test_split(total_normalised)

    # Split data into x and y
    x_train, y_train = xy_split(train)
    x_test, y_test = xy_split(test)

    # From Pandas to Numpy
    x_train = x_train.values
    x_test = x_test.values

    y_train = y_train.values
    y_test = y_test.values

    return (x_train, y_train, x_test, y_test, id)

def xy_split(train):
    x = train[['Weekday', 'Month', 'AWND', 'PRCP', 'SNOW', 'TAVG', 'TMAX', 'TMIN']]#, 'Holiday']]
    y = train[['BikeRides']]

    return (x, y)


def load_csv(filename):
    '''Reads the input and returns train/test split dataframes.
       Both dataframes will have N feature columns followed by a column labeled
       y.'''
    input_df = pd.read_csv(filename)

    return input_df
