"""
This program will predict the 30 day close price with open and close percentage change, high and low percentage change
"""

from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import time
from matplotlib import style
import matplotlib.pyplot as plt
import math
import datetime
import numpy as np
import pickle
from sklearn import preprocessing


df = pd.read_csv('./Data/yahoo.csv')
df = df.set_index('Date')
style.use('ggplot')

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
print("new_df is: --->", new_df)
new_df = new_df[['Close','HighVsLow_change','OpenVsClose_change','Volume']]

print (new_df)

# recording the real Closing price before testing for the forecasting the Price
original_DataFrame =pd.DataFrame(columns=['Date','Price'])


original_DataFrame =pd.DataFrame(new_df['Close'].values , columns=['Price'])

# there are 1098 tuples, - 30 is 1067
original_DataFrame = original_DataFrame[1068:1098] # This Before the 30 day of the end day
#print ("original_DataFrame is ----->", original_DataFrame)


#original_DataFrame = pd.DataFrame({'Date': next_date_array[:],'Price': Forecast_set[:]})
plt.plot (original_DataFrame['Price'])

plt.xlabel("Date")
plt.ylabel("Price")
plt.show()


#  # will forcast the Closing price
#
forecast_col = 'Close'
# # # get the data set length percentage's 0.1 will be in the forecasted
forecast_out = int(math.ceil(0.027* len(new_df)))# 30 days of the data out of 1098 days , accurency with 75% to 83% swinging, +-8% change
#print ("Forcast_out is : ", forecast_out)
#
 # preparaing for the empty labels for the incoming forcast
#
new_df['label']= new_df[forecast_col].shift(-forecast_out)

#print (new_df['label'])

# get x value and y value of as rest of the data column and label column
#
X = np.array(new_df.drop(['label'],1)) # all columns, other than label column
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]
new_df.dropna(inplace=True)
y = np.array(new_df['label'])# only label column


#print (len(X), len(y))
# # # get testing set and training set
# #
# # #split the dataset with a random seed
# # # training size is the 90% of the data set
# #
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# # Linear regression model
# linear_regression = LinearRegression()
# linear_regression.fit(x_train,y_train)
# with open ('./Data/linearregressionFitted.pickle', 'wb') as f:
#      pickle.dump(linear_regression,f)



pickled = open('./Data/linearregressionFitted.pickle','rb')
linear_regression = pickle.load(pickled)

# # not to come out the negative value in the accurency score
LinearRegression(copy_X= True, fit_intercept=True,n_jobs=1, normalize=False)
accuracy = linear_regression.score(x_test,y_test)
# #
print ('\nLinear_regression accuracy is :', accuracy)

svm_modle= svm.SVR()
svm_modle.fit(x_train,y_train)


accuracy_SVR = svm_modle.score(x_test,y_test)
print ("svR_accurency:", accuracy_SVR)
#

 #predit the stock price for the bitcoin for next 0.1% of the day which is 4 day for here
Forecast_set = linear_regression.predict(X_lately)

print ("Forecast_set is :", Forecast_set)
new_df['Forecast'] = np.nan

last_date = new_df.iloc[-1].name

# print ("last date is: " , last_date)
last_date = time.mktime(datetime.datetime.strptime(last_date,"%Y-%m-%d").timetuple())
# print ("timestamp is: ",last_date)
#
#
#
one_day = 86400
next_unix = last_date + 86400
# print ("next unix is: ", next_unix)

## just to show the forcast_set with Price values
label_arry = np.array(new_df['label'])


next_date_array= []
# # just visualization ( later get inside the
for i in Forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)# might be this one wrong
    next_date = str(next_date)
    #print ("String next date is:", next_date)
    next_date= str.split(next_date," ")
    #print (("Splited string next date is:", next_date[0]))
    next_date = next_date[0]
    next_date_array.append(next_date) # just to get an arrray for later use


    # this should be in function (change it later)
    next_unix +=one_day
    new_df.loc[next_date] = [np.nan for _ in range(len(new_df.columns)-1)] +[i]


print ("next_date array is: ", next_date_array)

# make a forcast vs nexdate dataset for Demo

newDf = pd.DataFrame(columns=['Date','Price'])



newDf = pd.DataFrame({'Date': next_date_array[:],'Price': Forecast_set[:]})
#original_Data= pd.DataFrame({'Date': next_date_array[:],'Price': original_DataFrame['Price']})
print ("newDf is: ---->", newDf)

# this just to get comparison price with Real Price

Forecast_DataFrame= pd.DataFrame(Forecast_set)
print ("Forecast DataFrame is ", Forecast_DataFrame)
Forecast_DataFrame.to_csv('./Data/Forecast.csv')
# let's do the same dataframe


# plt.plot(newDf['Price'])
#
# plt.title("Forcasted Price vs Original Price graph")
# plt.xlabel("Date")
# plt.ylabel("Prices")
# plt.show()
# print (len(next_date_array))
# print (len(Forecast_set))

#new_df['Date']= next_date_array
#new_df['Price']= Forecast_set
# print (new_df)



#------------------------------------------------------
#new_df['Close'].plot()
new_df['Forecast'].plot()

plt.title("January 1st, 2016 To January 1st, 2019 bitcoin Stock price and 30 day forecast")
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
#plt.show()

