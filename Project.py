# -*- coding: utf-8 -*-
"""
Created on Sun May 10 22:18:24 2020

@author: PROSSENJIT
"""

import pandas as pd
import numpy as np
dataset = pd.read_csv('train.csv')
dataset.mean()
dataset.describe()

fill_na = ['LotFrontage']
empty_column = ['MSZoning', 'Utilities', 'Condition2', 'HouseStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'BsmtFinSF2', 'Heating', 'Electrical', 'LowQualFinSF', 'KitchenQual', 'Functional', 'GarageYrBlt', 'GarageQual', 'PoolQC', 'MiscFeature', 'MiscVal', 'SaleType','ScreenPorch','GarageYrBlt', 'EnclosedPorch', '3SsnPorch', 'PoolArea','MasVnrArea']

dataset['LotFrontage'].fillna(value = dataset['LotFrontage'].mean(), inplace = True)
dataset.drop(empty_column, axis = 1, inplace = True)
dataset.dropna(axis = 0, inplace = True)
dataset.drop('Id', axis = 1, inplace = True)


dataset = pd.get_dummies(dataset, drop_first = True)
x = dataset.drop('SalePrice', axis = 1)
y = dataset['SalePrice']
dataset.columns


#Spliting into training and testing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2 , random_state = 0 )

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)
#Fitting linear regression model
from sklearn.linear_model import LinearRegression 
regression = LinearRegression(fit_intercept = True, normalize = False)
model = regression.fit(x_train , y_train)
model.score(x_train , y_train)


#Evaluation
from sklearn.metrics import mean_squared_error
y_pred = model.predict(sc_x.fit_transform(x_test))
mean_squared_error(y_test, y_pred)
import matplotlib.pyplot as plt
plt.scatter(x = y_pred, y = y_test)
plt.show()


#Applying SVR
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf', degree = 3)
regressor.fit(x_train , y_train)
y_pred = regressor.predict(x_test)
#Evaluation
from sklearn.metrics import mean_squared_error
y_pred = model.predict(sc_x.fit_transform(x_test))
mean_squared_error(y_test, y_pred)

#Applying random forest
#fitting Random forest on the data set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 500, criterion = 'mse', random_state =0)
model = regressor.fit(x_train,y_train)
y_pred = model.predict(x_test)
#Evaluation
from sklearn.metrics import mean_squared_error
y_pred = model.predict(x_test)
mean_squared_error(y_test, y_pred)
import matplotlib.pyplot as plt
plt.scatter(x = y_pred, y = y_test)
plt.show()
y_pred_test_data = model.predict(test_data)

#model by feature selection
from sklearn.feature_selection import RFE
regression = LinearRegression(fit_intercept = True, normalize = False)
model = RFE(regression, n_features_to_select = 100, step = 1)
model = model.fit(x_train , y_train)
model.ranking_
model.support_
model.get_params()
model.predict(x_train)

#Evaluation
from sklearn.metrics import mean_squared_error
y_pred_train = model.predict(x_train)
mean_squared_error(y_train, model.predict(x_train))

import matplotlib.pyplot as plt
plt.scatter(x = model.predict(x_train), y = y_train)
plt.show()

#Statsmodels apply
import statsmodels.api as sm
x = np.append(arr = np.ones((1370,1)).astype(int), values = x, axis = 1)
#Spliting into training and testing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2 , random_state = 0 )


x_optimum = x_train[:, 0:206]
regression_ols = sm.OLS(endog = y_train, exog = x_optimum)
model = regression_ols.fit()
model.summary()


def Backward_selection(x_OPTM,pvalue):
    while True : 
        regression_OLS = sm.OLS(endog = y_train, exog = x_OPTM).fit()
        for i in range(0,len(regression_OLS.pvalues.astype(float))) :
            if (regression_OLS.pvalues.astype(float)[i]) > pvalue :
                x_OPTM = np.delete(x_OPTM,i,1)
                break
        else:
            return regression_OLS.summary()
            return x_OPTM

Backward_selection(x_optimum, 0.05)

#Tensorflow approach
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(output_dim = 103, input_dim = 206, kernel_initializer='normal', activation = 'relu'))
model.add(Dense(output_dim = 103, kernel_initializer='normal', activation = 'relu'))
model.add(Dense(output_dim = 1, activation = 'linear'))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x_train, y_train,batch_size = 100, epochs = 100)

y_pred = model.predict(x_test)

import matplotlib.pyplot as plt
plt.scatter(x = y_pred, y = y_test)
plt.show()



