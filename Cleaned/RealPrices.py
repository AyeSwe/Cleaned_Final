"""
This program will analyse the short term predition with the last 3 day of price
"""


import datetime

import datetime as dt
import pandas as pd
import pandas_datareader.data as web
from matplotlib import style
import matplotlib.pyplot as plt
import math
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.metrics import accuracy_score

symbol= 'BTC-USD'

def got_start_Date():
    return datetime.datetime.now() + datetime.timedelta(-30)



def got_end_Date():
     return dt.datetime.today()


#


df_Forcast = pd.read_csv('./Data/Forecast.csv',parse_dates=True,index_col=0 )
df_Forcast = df_Forcast.round(4)
df_Forcast.to_csv('./Data/Forecast.csv')

df_Forcast = pd.read_csv('./Data/Forecast.csv')

#print (df_Forcast)
df_Forcast.rename({"Unnamed: 0":"a"}, axis="columns", inplace=True)
df_Forcast.drop(["a"], axis=1, inplace=True)
#print ("df Forcast is",df_Forcast)

df = pd.read_csv('./Data/.csv',parse_dates=True,index_col=0 )
df = df.round(4)
df.to_csv('./Data/RealPrice.csv')
df = pd.read_csv('./Data/RealPrice.csv')
df = df.set_index('Date')
#print (df.info())
style.use('ggplot')







#----------------------

# df.plot(kind= 'box',subplots=True, layout= (1,6),sharex=False,sharey=False)
# df.hist()
# scatter_matrix(df)
# plt.show()
#df.rename(column= {})
# percentage change for open vs close, and high vs low

df ['OpenVsClose_change'] = (df['Close']-df['Open'])/ df['Open'] * 100
#
df ['HighVsLow_change'] = (df['High']-df['Low'])/ df['Low'] * 100
#
# # just change the data set
new_df = pd.DataFrame(df)
new_df = new_df[['Close','HighVsLow_change','OpenVsClose_change','Volume']]
#print("new_df is: --->", new_df)

# just the Closing column

close_Price = np.array(new_df['Close'])
forecast_price = np.array((df_Forcast.iloc[:,0]))


#--------------

# # red dashes, blue squares and green triangles
plt.plot(close_Price,'r--')
plt.show()










