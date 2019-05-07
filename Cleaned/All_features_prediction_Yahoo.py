"""
This program will predit the 30 days stock close price with all of the features (Open, high, low, Volume)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn import svm
from sklearn.linear_model import LinearRegression
import seaborn as sns; sns.set(font_scale = 1.2)

yahoo= pd.read_csv('./Data/yahoo.csv')
# testing for dropping date column, Adj Close column
#df_Forcast.drop(["a"], axis=1, inplace=True)
yahoo.drop(['Date','Adj Close'], axis =1, inplace= True)
yahoo['close']= yahoo['Close']
yahoo.drop(['Close'], axis =1, inplace= True)
yahoo = yahoo.dropna()
#print(yahoo.info())
#print (yahoo.tail())




#split the dat to training data and testing data

x = yahoo.iloc[:,: -1]# evey row and every column other than the last column
print (x.shape)
y = yahoo.iloc[:,-1]# only the last column : this is class data
print (y.shape)
x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(x,y, random_state=17, train_size=0.975)# predit about 10 day
print ("x train is: ",x_train_data)
print ("y_train is: ",y_train_data)
#
# # support vector machine model
#

#
#
# # Linear Regression model
#
linearRegression_model = LinearRegression()
#
linearRegression_model.fit(x_train_data, y_train_data)
print ("X tes data is: ", x_test_data)

x_array = np.array(x_test_data)
print ('\n x test array is :', x_array)


# This data will live streamed from the yahoo finiance later for demostration

# Open	High	Low	Close*	Adj Close**	Volume

# 5,753.38	5,901.36	5,596.15
# 5,840.08	5,840.08	263,390,273
# May 04, 2019
# Now predit with yesterday price
predit_May4th = linearRegression_model.predict([[5901.36,5596.15,5753.38,263390273]])
yesterday_index=[[5901.36,5596.15,5753.38,263390273]]

predit_linearRegression = linearRegression_model.predict(x_test_data)
array = np.array(predit_May4th)
#print ('\n Testing the model with these feature for the closing price data is :\n' , x_test_data)

#print ('\n Testing the model with yesterday price :\n' ,yesterday_index )


#print ('count is \n:', array.size)
#print("Linear Regression model predition is : ",predit_May4th)


#Here prediction of 30 days accurency
accurency_linear = linearRegression_model.score(x_test_data, y_test_data)
#
print("Confident  of LinearRegression model is: ", accurency_linear)



