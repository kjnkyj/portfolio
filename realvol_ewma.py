import pandas as pd
import numpy as np
import datetime

def ETL(data = input()):
    ## will be importing and formatting data
    df = pd.read_csv(data)
    df.created_date = pd.to_datetime(df.created_date)
    df = df.iloc[:, 1:]
    df_mar19 = df[df.expiry == '19-Mar']
    df_jun19 = df[df.expiry == '19-Jun']

    min_mar19 = df_mar19.groupby(df_mar19.created_date.dt.floor('5T')).tail(n=1)
    min_mar19 = min_mar19.reset_index()
    min_mar19['index'] = min_mar19.index
    min_mar19 = min_mar19.set_index('created_date')
    ## resort for calculating return
    min_mar19 = min_mar19.sort_index()

    df_jun19 = df[df.expiry == '19-Jun']
    min_jun19 = df_jun19.groupby(df_jun19.created_date.dt.floor('5T')).tail(n=1)
    min_jun19 = min_jun19.reset_index()
    min_jun19['index'] = min_jun19.index
    min_jun19 = min_jun19.set_index('created_date')
    ## resort for calculating return
    min_jun19 = min_jun19.sort_index()
    for i in [min_jun19, min_mar19]:
        log_ret(i)

    return min_mar19 , min_jun19

def log_ret(df):
    df = df.sort_index()
    df['log_ret'] = np.log(df.price) - np.log(df.price.shift(1))
    df['log_ret_sqr'] = np.power(df['log_ret'],2)
    df = df.sort_index(ascending = False)

def x_weight_lambda(lag, pct = 0.5):
    try:
        type(lag) == int
    except:
        print 'Variable Lag is not an Integer'
        return 'Variable Lag is not an Integer'

    polynom = [-1] + [0 for i in range(lag)] + [1-pct]
    p = np.poly1d(polynom)
    print(np.poly1d(p))
    ewma_lambda = max([i for i in p.r.real])
    if ewma_lambda > 1:
        return 'error'
    else:
        return ewma_lambda

def dates_rows_count(df,days_num, offset = None):
    ###add filter on offset, minimum scale for calculation
    return len(df.loc[: df.index.date[0] - pd.Timedelta(days = days_num)])


def weighted_log_ret_sqr(df,ewma_lambda):
    try:
        df[['index','log_ret_sqr']].head()
    except ValueError:
        print "you don't have either index or log ret sqr"


    df['weight'] = ((1-ewma_lambda)*np.power(ewma_lambda,df['index']))
    df['weighted_log_ret_sqr'] = df['weight'] * df['log_ret_sqr']


def main():
    mar19 , jun19 = ETL(input())
    print 'day3 50% weight for mar-19'
    print x_weight_lambda(dates_rows_count(mar19, 1))


