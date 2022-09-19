

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats




def nulls_by_row(df):
    num_missing = df.isnull().sum(axis=1)
    prnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame(
        {'num_cols_missing': num_missing, 
        'percent_cols_missing': prnt_miss})\
    .reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).\
    count().reset_index().rename(
        columns={
            'customer_id': 'count'})
    return rows_missing

def nulls_by_col(df):
    num_missing = df.isnull().sum()
    percnt_miss = num_missing / df.shape[0] * 100
    cols_missing = pd.DataFrame(
    {
        'num_rows_missing': num_missing,
        'percent_rows_missing': percnt_miss
    })
    return  cols_missing


def summarize(df):
    print('DataFrame head: \n')
    print(df.head())
    print('----------')
    print('DataFrame info: \n')
    print(df.info())
    print('----------')
    print('Dataframe Description: \n')
    print(df.describe())
    print('----------')
    print('Null value assessments: \n')
    print('nulls by column: ', nulls_by_col(df))
    print('--')
    print('nulls by row: ', nulls_by_row(df))
    numerical_cols = [col for col in df.columns if df[col].dtype != 'O']
    categorical_cols = [col for col in df.columns if col not in numerical_cols ]
    print('--------')
    print('value_counts: \n')
    for col in df.columns:
        print('Column Name: ', col)
        if col in categorical_cols:
            print(df[col].value_counts())
        else:
            print(df[col].value_counts(bins=10, sort=False))
        print('--')
    print('---------')
    print('Report Finished')





# for some series s:
def get_upper_outliers(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + k*iqr
    return s.apply(lambda x: max([x - upper_bound, 0]))




def add_upper_outlier_columns(df, k=1.5):
    for col in df.select_dtypes('number'):
        df[col + '_upper_outliers'] = get_upper_outliers(df[col], k)
    return df


def remove_columns(df, cols_to_remove):
    df = df.drop(columns=cols_to_remove)
    return df

def handle_missing_values(df, 
                          prop_required_columns=0.5, 
                          prop_required_row=0.75):
    threshold = int(round(prop_required_columns * len(df.index), 0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=threshold)
    return df  
