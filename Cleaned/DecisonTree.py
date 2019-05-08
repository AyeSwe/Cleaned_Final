"""
This program will predit the 30 days stock close price with all of the features (Open, high, low, Volume)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split
import seaborn as sns; sns.set(font_scale = 1.2)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor as DecisonTree
from sklearn.metrics import mean_squared_error


yahoo= pd.read_csv('./Data/yahoo.csv')
# testing for dropping date column, Adj Close column
#df_Forcast.drop(["a"], axis=1, inplace=True)
yahoo.drop(['Date','Adj Close'], axis =1, inplace= True)
yahoo['close']= yahoo['Close']
yahoo.drop(['Close'], axis =1, inplace= True)
yahoo = yahoo.dropna()
print ("Data summery: ", yahoo.describe())

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

# normalize the x train data and x testing data
scale = StandardScaler()
xtrain_scale= scale.fit_transform(x_train_data)
xtest_scale = scale.transform(x_test_data)

#
# # Decision Tree model
#
D_Tree_model = DecisonTree(max_depth=2) # to avoid outfitting max_depth is controlled.
#
# # train with scaled x train data and y train data
D_Tree_model.fit(xtrain_scale, y_train_data)
#
#
tree_predit = D_Tree_model.predict(xtest_scale)
print ("D_Tree Prediction is ", tree_predit)

D_treeAccurecy = D_Tree_model.score(xtest_scale,y_test_data)
#linearRegression_model.score(x_test_data, y_test_data)

print ("Decision Tree model accurency is: ", D_treeAccurecy )
#
# # Evaluation of Decision tree model
#
DT_MeanError = mean_squared_error(y_train_data,D_Tree_model.predict(xtrain_scale))
Root_mean_square = sqrt(DT_MeanError)
#
#print("Decision Tree training mse = ",tree_mse," & mae = ",tree_mae," & rmse = ", sqrt(tree_mse))
print ("Mean square error the decision tree model", DT_MeanError)
#
# Now see how the behavior or mean square error on the test data.
#tree_test_mse = mean_squared_error(y_test, tree_model.predict(test_scaled))

D_treeTestMeanError = mean_squared_error(y_test_data, D_Tree_model.predict(xtest_scale))
Root_mean_square = sqrt(D_treeTestMeanError)
print ("Testing the testing data to see how model generalize the prediction: ", Root_mean_square)
tree_predit_array = np.array(tree_predit)
y_test_data_array = np.array(y_test_data)


# sending to csv for furter comparison


newDf = pd.DataFrame(columns=['Close'])
newDf['Close']= tree_predit_array # fill the column with Date

newDf.to_csv('./Data/DecisionTreePrediction.csv')

plt.plot(y_test_data_array)
plt.plot(tree_predit_array )
plt.title(" 30 day forecast with Decision Tree Regression price and Real Price ")
# plt.legend()
# plt.show()
graph = plt.subplot(111)
box = graph.get_position()
graph.set_position([box.x0, box.y0, box.width*0.65, box.height])
legend_x = 1
legend_y = 0.5
plt.legend(["Predicted Price","Real Price" ], loc='center left', bbox_to_anchor=(legend_x, legend_y))
plt.show()