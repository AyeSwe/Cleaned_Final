"""
This program will take the linearRegression model fit as pickle from both of the API dataset
"""



from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib import style
import math
import datetime
import numpy as np
import pickle


df = pd.read_csv('./Data/yahoo.csv')
#print (df.info())
df = df.set_index('Date')
#print (df.head())
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
new_df = new_df[['Close','HighVsLow_change','OpenVsClose_change','Volume']]
#print("new_df is: --->", new_df)





# recording the real Closing price before testing for the forecasting the Price
original_DataFrame =pd.DataFrame(columns=['Date','Price'])


original_DataFrame =pd.DataFrame(new_df['Close'].values , columns=['Price'])

# there are 1098 tuples, - 30 is 1067
original_DataFrame = original_DataFrame[1068:1098] # This Before the 30 day of the end day
#print ("original_DataFrame is ----->", original_DataFrame)


#original_DataFrame = pd.DataFrame({'Date': next_date_array[:],'Price': Forecast_set[:]})




#  # will forcast the Closing price
#
forecast_col = 'Close'
# # # get the data set length percentage's 0.1 will be in the forecasted
forecast_out = int(math.ceil(0.027* len(new_df)))# 30 days of the data out of 1098 days , accurency with 75% to 83% swinging, +-8% change
#print ("Forcast_out is : ", forecast_out)
#
# # # preparaing for the empty labels for the incoming forcast
#
new_df['label']= new_df[forecast_col].shift(-forecast_out)

#print (new_df['label'])

# # # get x value and y value of as rest of the data column and label column
#
X = np.array(new_df.drop(['label'],1)) # all columns, other than label column
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


# Linear regression model
linear_regression = LinearRegression(n_jobs = -1)
linear_regression.fit(x_train,y_train)
with open ('./Data/linearregressionFitted.pickle', 'wb') as f:
    pickle.dump(linear_regression,f)

# pickled = open('./Data/linearregressionFitted.pickle','rb')
# linear_regression = pickle.load(pickled)

# # not to come out the negative value in the accurency score
LinearRegression(copy_X= True, fit_intercept=True,n_jobs=1, normalize=False)
accuracy = linear_regression.score(x_test,y_test)
# #
print ('\nLinear_regression accuracy is :', accuracy)

