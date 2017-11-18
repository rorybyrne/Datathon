from sklearn import preprocessing
import pandas as pd
from pprint import pprint

scaler = preprocessing.MinMaxScaler()

def fix_nan(df):
    # val_cols = ['AWND', 'PRCP', 'SNOW', 'TAVG', 'TMAX', 'TMIN']
    # return df.fillna(df.mean())

    return df.dropna()

def remove_negs(df, col):
    df = df.ix[df[col] > 0]
    return df

# def swap_pcrp(df):
#     for i, r in df.iterrows():
#         if(r['month'] == 1):
#             r['']

def process(df):
    # Select cols to normalise
    total = df

    # total = swap_prcp(total)

    # total = remove_negs(total, "SNOW")
    total = fix_outliers(total, 'TMIN', 0.25, 0.75)
    total = fix_outliers(total, 'TMAX', 0.25, 0.75)
    total = remove_negs(total, 'AWND')
    total = fix_outliers(total, 'TAVG', 0.25, 0.75)
    # total = fix_outliers(total, 'SNOW', 0.25, 0.75)
    # total = fix_outliers(total, 'PRCP', 0.25, 0.75)

    total = fix_nan(total)

    # to_normalise = total[['AWND', 'PRCP', 'SNOW', 'TAVG', 'TMAX', 'TMIN']]

    # x = to_normalise.values
    # x_scaled = scaler.fit_transform(x)

    # normalised_cols = to_normalise #pd.DataFrame(x_scaled, columns=to_normalise.columns)

    # Select non-normalise cols
    # not_normalise = nan_fix[['Weekday', 'Month', 'BikeRides', 'Holiday']].reset_index()
    # print("not_normalise shape: (%s, %s)" % not_normalise.shape)
    # pprint(not_normalise)
    #
    # total_normalised = not_normalise.join(normalised_cols)
    # print("Total normalised shape: (%s, %s)" % total_normalised.shape)

    return total

def fix_outliers(df_in, col_name, q1, q3):
    # q = df_in[col_name].quantile(0.90)
    # return df_in[df_in[col_name] < q]

    # print(df_in[col_name])

    print(col_name)

    q1 = df_in[col_name].quantile(q1)
    print(q1)
    q3 = df_in[col_name].quantile(q3)
    print(q3)
    iqr = q3 - q1  # Interquartile range
    fence_low = q1 - 1.5 * iqr
    print("fence_low: %s" % fence_low)
    fence_high = q3 + 1.5 * iqr
    print("fence_high: %s" % fence_high)
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out


def get_one_hot(df, col_name):
    '''Takes a col name and a df and deletes that col and appends a one hot
    matrix of it encoded.'''
    dummies = pd.get_dummies(df[col_name], prefix=col_name)
    df = df.join(dummies)
    df = df.drop(col_name, 1)

    return df
