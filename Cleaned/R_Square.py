"""
This program will analysis the linearRegression model that is fitted to the 3 year data set of the yahoo finican bitcoin price
with intercept, coefficients, and root mean error.

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns; sns.set(font_scale = 1.2)
from sklearn import metrics

yahoo= pd.read_csv('./Data/yahoo.csv')

# testing for dropping date column, Adj Close column
#df_Forcast.drop(["a"], axis=1, inplace=True)

# volume is dropped to see if R value is go down
yahoo.drop(['Date','Adj Close'], axis =1, inplace= True)
yahoo['close']= yahoo['Close']
yahoo.drop(['Close'], axis =1, inplace= True)
yahoo = yahoo.dropna()
#print(yahoo.info())
print (yahoo.tail())


x = yahoo.iloc[:,: -1]# evey row and every column other than the last column
print (x.shape)
y = yahoo.iloc[:,-1]# only the last column : this is class data

# get the testing data out of x and y
x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(x,y, random_state=17, train_size=0.975)# predit about 10 day
#  # Linear Regression model
# # #
linearRegression_model = LinearRegression()
# # training the High feature and close price
linearRegression_model.fit(x_train_data,y_train_data)

# getting the intercept point from
print ("Intercept: ", linearRegression_model.intercept_)
print ("Coefficienc", linearRegression_model.coef_)

y_prediction = linearRegression_model.predict(x_test_data)
print ('close prediction: ', y_prediction)

# # plotting the least Squares Line
#
# sns.pairplot(yahoo, x_vars=['High','Low','Open','Volume'], y_vars='close', size=4, aspect=0.7, kind='reg')
# #plt.show()
#
print ('linearRegression_model.score() is :', linearRegression_model.score(x_train_data,y_train_data) )
#
# # Model Evaluation Metric for LinearRegression
# # MSE (Mean Squared error checking for features)
print ("Root Mean Error is ",np.sqrt(metrics.mean_squared_error(y_test_data, y_prediction)))
#
print ("Test data is : ", y_test_data)
realPrice = np.array(y_test_data)
print ("Test data is : ", realPrice)
#new_df['Close'].plot()
plt.plot(y_prediction)
plt.plot(realPrice)
plt.title(" 30 day forecast price and Real Price")
# plt.legend()
# plt.show()
graph = plt.subplot(111)
box = graph.get_position()
graph.set_position([box.x0, box.y0, box.width*0.65, box.height])
legend_x = 1
legend_y = 0.5
plt.legend(["Real Price", "Predicted Price"], loc='center left', bbox_to_anchor=(legend_x, legend_y))
plt.show()

