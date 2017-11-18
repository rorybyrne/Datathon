from sklearn import preprocessing
import pandas as pd
from pprint import pprint

scaler = preprocessing.MinMaxScaler()

def preprocess(df):
    df = drop_nan(df)

    return df

def drop_nan(df):
    return df.dropna()

def normalise(df):
    # return df
    x = df.values
    x_scaled = scaler.fit_transform(x)

    out_df = pd.DataFrame(x_scaled, columns=df.columns)
    print("Normalized Cols")
    pprint(out_df)

    return out_df