import pandas as pd


def train_test_split(df, train_percent=0.8):
    '''Randomly samples and splits data into train/test split.'''
    # Uncomment these lines to maintain ordering:
    #split_index = int(df.shape[0] * train_percent)
    #return (df[:split_index], df[split_index:])

    train = df.sample(frac=0.8, random_state=200)
    test = df.drop(train.index)

    return (train, test)


def get_input():
    '''Reads the input and returns train/test split dataframes.
       Both dataframes will have N feature columns followed by a column labeled
       y.'''
    input_df = pd.read_csv('../data/test_input.csv')

    # DO PREPROCESSING HERE:

    train_set, test_set = train_test_split(input_df)
    
    return (train_set, test_set)
