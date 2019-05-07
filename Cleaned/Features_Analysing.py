"""
This program analysis the yahoo finiance dataset
features with LinearRegression model, intercepts, coefficients, and mean square of each features
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns; sns.set(font_scale = 1.2)
from sklearn import metrics

yahoo= pd.read_csv('./Data/yahoo.csv')



yahoo.drop(['Date','Adj Close'], axis =1, inplace= True)
yahoo['close']= yahoo['Close']
yahoo.drop(['Close'], axis =1, inplace= True)
yahoo = yahoo.dropna()
#print(yahoo.info())
print (yahoo.tail())



# this will do the linear regression performance for each features

feature_high = ['High'] # High, Low, Open, Volume,
x = yahoo[feature_high]
y = yahoo.iloc[:,-1] # target label

# get the testing data out of x and y
x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(x,y, random_state=17, train_size=0.975)# predit about 10 day
 # Linear Regression model
# #
linearRegression_model = LinearRegression()
# # training the High feature and close price
linearRegression_model.fit(x_train_data,y_train_data)

# getting the intercept point from
print (linearRegression_model.intercept_)
print (linearRegression_model.coef_)

# now predit clos price with the new High price (5901.36)
# make new dataframe

#new_High_data=pd.DataFrame({'High': [5901.36]})
#new_High = linearRegression_model.predict(new_High_data)
y_prediction = linearRegression_model.predict(x_test_data)
print ('close prediction: ', y_prediction)

# plotting the least Squares Line

sns.pairplot(yahoo, x_vars=['High','Low','Open','Volume'], y_vars='close', size=4, aspect=0.7, kind='reg')
plt.show()

print ('linearRegression_model.score() is :', linearRegression_model.score(x_train_data,y_train_data) )

# Model Evaluation Metric for LinearRegression
# MSE (Mean Squared error checking for features)
print("Root Mean Error is ",np.sqrt(metrics.mean_squared_error(y_test_data, y_prediction)))