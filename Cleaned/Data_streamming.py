"""
#   This program live stream the bitcoin price from teh coindesk API
#   3 years of daily stock information are gathered
#   perform some necessary data prepartion before saving to the further analysis
#   streamed data are saved as coindesk.csv
"""


"""
This program will live stream from the yahoo finical anaylis of bitcoin value in US dollar
- use the LinearRegression algorithm to predit the next 10 days of the stock
- other anaylis like stock aggresionness with volume and open and close percentage change
- if time permit , PCA will use the reduction of the dimension inorder to get the more accurency %
- try again with SVM also if time permit
- will use the every 4 repetation of 6 months ( saved data) worth data ( predit, 1 month) for perdition from 2017 January to June,  July to December, 2018 Janunary to Jue,July to Decemeber,
2019 April 1st to April 30  (May 1st to 9th predition) ( live Stream)
- how to show result?
- visualization
- use lineregression graph to compare the predicted data and real data (from the same website, same data as prediction)
- show accurency rate for the predicted data compare with algorithm accurency vs real data predition

"""


import matplotlib.pyplot as plt
import requests
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib import style
import matplotlib.pyplot as plt
import math
import numpy as np
import time
import datetime
import pickle
from sklearn import svm
import quandl
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import datetime as dt
import pandas as pd
import pandas_datareader.data as web
from matplotlib import style
import matplotlib.pyplot as plt
import math
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.metrics import accuracy_score




r = requests.get('https://api.coindesk.com/v1/bpi/historical/close.json?start=2016-01-01&end=2019-01-02').json()
#print(r)
#sending jason file into the dat frame
df = pd.DataFrame(r, columns=['bpi'])
#df.to_csv('./Data/coindesk.csv')
#print (df.info())
# drop the null values
df.dropna(inplace=True)

#split a cloumn to two column

#since "bpi" is one columns with values and key dictionary type

# sending values of the bpi as price
Price = df['bpi'].values
#print (Price)

# sending key (Dates) of the bpi as Date
Date= df['bpi'].keys()
#print (Date)

# making a new data frame with columns labels
newDf = pd.DataFrame(columns=['Date','Close'])
newDf['Date']= Date # fill the column with Date
newDf['Close']= Price# fill the column with Prices

newDf.to_csv('./Data/coindesk.csv')

#----------------------Stramming of coinDesk data end here and yahoo finiciance data streamming start here----------------------


symbol= 'BTC-USD'

start= dt.datetime(2016,1,1)
end = dt.datetime.now()
df = web.DataReader(symbol, 'yahoo', start, end)
df.to_csv('./Data/yahoo.csv')
#print (df)
df = pd.read_csv('./Data/yahoo.csv',parse_dates=True,index_col=0 )
df = df.round(4)
df.to_csv('./Data/yahoo.csv')

