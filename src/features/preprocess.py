from sklearn import preprocessing
import pandas as pd
from pprint import pprint

scaler = preprocessing.MinMaxScaler()

def preprocess(df):
    df = drop_nan(df)

    return df

def drop_nan(df):
    return df.dropna()

def process(df):
    # Select cols to normalise
    total = df
    # total = remove_outliers(total, 'TMIN')
    # total = remove_outliers(total, 'TMAX')
    total = remove_outliers(total, 'TAVG')
    # total = remove_outliers(total, 'PRCP')

    to_normalise = total[['AWND', 'PRCP', 'SNOW', 'TAVG', 'TMAX', 'TMIN']]

    x = to_normalise.values
    x_scaled = scaler.fit_transform(x)

    normalised_cols = pd.DataFrame(x_scaled, columns=to_normalise.columns)

    # Select non-normalise cols
    not_normalise = total[['Weekday', 'Month', 'BikeRides', 'Holiday']].reset_index()
    print("not_normalise shape: (%s, %s)" % not_normalise.shape)
    pprint(not_normalise)

    total_normalised = not_normalise.join(normalised_cols)
    print("Total normalised shape: (%s, %s)" % total_normalised.shape)

    return total_normalised

def remove_outliers(df_in, col_name):
    # q = df_in[col_name].quantile(0.90)
    # return df_in[df_in[col_name] < q]

    q1 = df_in[col_name].quantile(0.1)
    q3 = df_in[col_name].quantile(0.9)
    iqr = q3 - q1  # Interquartile range
    fence_low = q1 - 1.5 * iqr
    fence_high = q3 + 1.5 * iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out


def get_one_hot(df, col_name):
    '''Takes a col name and a df and deletes that col and appends a one hot
    matrix of it encoded.'''
    dummies = pd.get_dummies(df[col_name], prefix=col_name)
    df = df.join(dummies)
    df = df.drop(col_name, 1)

    return df
