import pandas as pd
import numpy as np
import datetime
import calendar
import os
#from __future__ import division

APP_DIRECTORY = os.path.abspath(os.path.dirname(__file__)) + '/'

FOMC = [datetime.date(2019,1,30),
      datetime.date(2018,12,19),
       datetime.date(2018,11,8),
       datetime.date(2018,9,26),
       datetime.date(2018,8,1),
       datetime.date(2018,6,13),
       datetime.date(2018,5,2),
       datetime.date(2018,3,21),
       datetime.date(2018,1,31)]
FOMC_hour = 'CDT 13:00'
## NFP datetime is for looped in a function

NFP_hour = 'CDT 7:00'


AAPL = [(datetime.date(2019,1,29),15),
       (datetime.date(2018,11,1),15),
       (datetime.date(2018,7,31),15),
       (datetime.date(2018,5,1),15),
       (datetime.date(2018,2,1),20)]
AAPL_date, AAPL_hour_int = list(zip(*AAPL))

MSFT = [(datetime.date(2019,1,30),15),
       (datetime.date(2018,10,24),15),
       (datetime.date(2018,7,19),15),
       (datetime.date(2018,4,26),15),
       (datetime.date(2018,1,31),20)]
MSFT_date, MSFT_hour_int = list(zip(*MSFT))

AMZN = [(datetime.date(2019,1,31),15),
       (datetime.date(2018,10,25),15),
       (datetime.date(2018,7,26),15),
       (datetime.date(2018,4,26),15),
       (datetime.date(2018,2,1),20)]
AMZN_date, AMZN_hour_int = list(zip(*AMZN))






def ETL(data = input()):
    ## will be importing and formatting data
    pd.read_csv(data)

    pass

def main(data):
    ## Assuming df is pandas and has datetime index
    df = ETL(data)
    df = df.sort_index()
    df['log_ret'] = np.log(df.price) - np.log(df.price.shift(1))
    main_filter = filter_outlier(df)
    df['hour_vol'] = df.log_ret
    df.loc[~main_filter,'hour_vol'] = None
    df['rolling_hour_vol'] = df.hour_vol.rolling('1H').std()
    df['rolling_hour_var'] = df.hour_vol.rolling('1H').var()
    b = df.groupby(df.index.hour).rolling_hour_var.mean()
    b.iloc[15] = b.iloc[15] * 1/6

    weight = (b)/sum(b)*100
    weight = weight.reset_index()
    weight.columns = ['hour','variance_weight']
    weight.to_csv(APP_DIRECTORY + 'hourly_variance_weight.csv',index = False)


def filter_outlier(df):
    ## returns filter

    hour_filter = (df.index.strftime('%H:%M:%S') != '17:00:00')

    ## NFP datetime
    NFP = []
    for year in [2018,2019]:
        for month in range(1,13):
            cal = calendar.monthcalendar(year,month)
            first_friday_date = cal[0][-3]
            if first_friday_date == 0:
                first_friday_date = cal[1][-3]
            NFP.append(datetime.date(year,month,first_friday_date))

    FOMC_filter = ((np.isin(df.index.date, FOMC)) & (df.index.strftime('%H:%M:%S') >= '12:30:00')
                   & (df.index.strftime('%H:%M:%S') <= '15:30:00'))
    NFP_filter = ((np.isin(df.index.date, NFP)) & (df.index.strftime('%H:%M:%S') >= '06:30:00')
                  & (df.index.strftime('%H:%M:%S') <= '09:30:00'))
    MSFT_filter = ((df.index.date == MSFT_date[0]) & (df.index.strftime('%H:%M:%S') >= '13:00:00')
                   & (df.index.strftime('%H:%M:%S') <= '17:00:00'))
    for i in range(1, len(MSFT)):
        MSFT_filter = MSFT_filter | ((df.index.date == MSFT_date[i])
                                     & (df.index.strftime('%H:%M:%S') >= '{}:00:00'.format(MSFT_hour_int[i] - 2))
                                     & (df.index.strftime('%H:%M:%S') <= '{}:00:00'.format(MSFT_hour_int[i] + 2)))

    AAPL_filter = ((df.index.date == AAPL_date[0]) & (df.index.strftime('%H:%M:%S') >= '13:00:00')
                   & (df.index.strftime('%H:%M:%S') <= '17:00:00'))
    for i in range(1, len(MSFT)):
        AAPL_filter = AAPL_filter | ((df.index.date == AAPL_date[i])
                                     & (df.index.strftime('%H:%M:%S') >= '{}:00:00'.format(AAPL_hour_int[i] - 2))
                                     & (df.index.strftime('%H:%M:%S') <= '{}:00:00'.format(AAPL_hour_int[i] + 2)))

    AMZN_filter = ((df.index.date == AMZN_date[0])
                   & (df.index.strftime('%H:%M:%S') >= '13:00:00') & (df.index.strftime('%H:%M:%S') <= '17:00:00'))
    for i in range(1, len(MSFT)):
        AMZN_filter = AMZN_filter | ((df.index.date == AMZN_date[i])
                                     & (df.index.strftime('%H:%M:%S') >= '{}:00:00'.format(AMZN_hour_int[i] - 2))
                                     & (df.index.strftime('%H:%M:%S') <= '{}:00:00'.format(AMZN_hour_int[i] + 2)))

    event_filter = (FOMC_filter | NFP_filter | MSFT_filter | AMZN_filter | AAPL_filter)

    std_3_filter = ~(abs(df.log_ret - df.log_ret.mean()) > 3 * df.log_ret.std())

    main_filter = (~event_filter & hour_filter) & std_3_filter

    return main_filter




if __name__ == "__main__":
    main()