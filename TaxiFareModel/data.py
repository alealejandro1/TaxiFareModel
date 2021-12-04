import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

import TaxiFareModel.params as params

AWS_BUCKET_PATH = "s3://wagon-public-datasets/taxi-fare-train.csv"

def get_data(nrows=10_000):
    '''returns a DataFrame with nrows from s3 bucket'''
    df = pd.read_csv(AWS_BUCKET_PATH, nrows=nrows)
    # df = pd.read_csv(f"gs://{params.GCLOUD_BUCKET_NAME}/{params.GCLOUD_TRAIN_DATA_PATH}",nrows=1000)
    return df

def df_optimized(df, verbose=True, **kwargs):
    """
    Reduces size of dataframe by downcasting numeircal columns
    :param df: input dataframe
    :param verbose: print size reduction if set to True
    :param kwargs:
    :return: df optimized
    """
    in_size = df.memory_usage(index=True).sum()
    # Optimized size here
    for type in ["float", "integer"]:
        l_cols = list(df.select_dtypes(include=type))
        for col in l_cols:
            df[col] = pd.to_numeric(df[col], downcast=type)
            if type == "float":
                df[col] = pd.to_numeric(df[col], downcast="integer")
    out_size = df.memory_usage(index=True).sum()
    ratio = (1 - round(out_size / in_size, 2)) * 100
    GB = out_size / 1000000000
    if verbose:
        print("optimized size by {} % | {} GB".format(ratio, GB))
    return df

def clean_data(df, test=False):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df

def get_Xy(df, test=False):
    X = df.drop(columns=['fare_amount'])
    y = df['fare_amount']
    return (X,y)

def holdout(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.1)
    return (X_train, X_test, y_train, y_test)


def data_preparation():
    df = get_data()
    df = clean_data(df)
    df = df_optimized(df)
    X, y = get_Xy(df)
    return holdout(X, y)
####
class DfOptimizer(BaseEstimator, TransformerMixin):
    """
    Reduces size of dataframe by downcasting numeircal columns
    :param df: input dataframe
    :param verbose: print size reduction if set to True
    :param kwargs:
    :return: df optimized
    """
    def __init__(self):
        self.verbose = True

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, **kwargs):
        df = pd.DataFrame(X)
        assert isinstance(df, pd.DataFrame)
        in_size = df.memory_usage(index=True).sum()
        # Optimized size here
        for type in ["float", "integer"]:
            l_cols = list(df.select_dtypes(include=type))
            for col in l_cols:
                df[col] = pd.to_numeric(df[col], downcast=type)
                if type == "float":
                    df[col] = pd.to_numeric(df[col], downcast="integer")
        out_size = df.memory_usage(index=True).sum()
        ratio = (1 - round(out_size / in_size, 2)) * 100
        GB = out_size / 1_000_000_000
        if self.verbose:
            print("optimized size by {} % | {} GB".format(ratio, GB))
        return df

###


if __name__ == '__main__':
    df = get_data()
