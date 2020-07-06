# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 01:58:13 2020

@author: PROSSENJIT
"""


#Importing the dataset
import pandas as pd
dataset = pd.read_csv('sales_train_validation.csv')
dataset.drop(['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], axis = 1, inplace = True)

dataset.head()
dataset.index

Hobbies_set = []
Household_set = []
Food_set = []

for i in range(len(dataset)):
    if 'HOBBIES' in dataset.iloc[i,0]:
        Hobbies_set.append(dataset.iloc[i,0])
    elif 'HOUSEHOLD' in dataset.iloc[i,0]:
        Household_set.append(dataset.iloc[i,0])
    else:
        Food_set.append(dataset.iloc[i,0])


df = dataset.transpose()
calender_dataset = pd.read_csv('calendar.csv', index_col = 'date', parse_dates = True)
header_row = 0
df.columns = df.iloc[header_row]
df.drop('id', inplace = True)
df['day'] = calender_dataset.index[0:1913]
df.head()
df.tail()
df.set_index('day', inplace = True)
df.index.freq = 'D'
df.index
df.columns
df.head()


#Classification based on mean
Hobbies_sub_0_025 = []
Hobbies_sub_025_05 = []
Hobbies_sub_05_075 = []
Hobbies_sub_075_1 = []
Hobbies_sub_1_2 = []
Hobbies_sub_2_inf = []
for i in range(len(Hobbies_set)):
    if df[Hobbies_set[i]].mean() >=0 and df[Hobbies_set[i]].mean()<0.25:
        Hobbies_sub_0_025.append(Hobbies_set[i])
    elif df[Hobbies_set[i]].mean() >=0.25 and df[Hobbies_set[i]].mean()<0.50:
        Hobbies_sub_025_05.append(Hobbies_set[i])
    elif df[Hobbies_set[i]].mean() >=0.50 and df[Hobbies_set[i]].mean()<0.75:
        Hobbies_sub_05_075.append(Hobbies_set[i])
    elif df[Hobbies_set[i]].mean() >=0.75 and df[Hobbies_set[i]].mean()<1:
        Hobbies_sub_075_1.append(Hobbies_set[i])
    elif df[Hobbies_set[i]].mean() >=1 and df[Hobbies_set[i]].mean()<2:
        Hobbies_sub_1_2.append(Hobbies_set[i])
    elif df[Hobbies_set[i]].mean() >=2 :
        Hobbies_sub_2_inf.append(Hobbies_set[i])
df[Food_set].plot(legend = True, figsize = (200,50)).legend(loc = 2,bbox_to_anchor = (1.0,1.0))
means_foods = df[Food_set].mean()
means_foods.min()
means_foods.max()
means_foods.value_counts()


Foods_sub_0_025 = []
Foods_sub_025_05 = []
Foods_sub_05_075 = []
Foods_sub_075_1 = []
Foods_sub_1_2 = []
Foods_sub_2_inf = []
for i in range(len(Food_set)):
    if df[Food_set[i]].mean() >=0 and df[Food_set[i]].mean()<0.25:
        Foods_sub_0_025.append(Food_set[i])
    elif df[Food_set[i]].mean() >=0.25 and df[Food_set[i]].mean()<0.50:
        Foods_sub_025_05.append(Food_set[i])
    elif df[Food_set[i]].mean() >=0.50 and df[Food_set[i]].mean()<0.75:
        Foods_sub_05_075.append(Food_set[i])
    elif df[Food_set[i]].mean() >=0.75 and df[Food_set[i]].mean()<1:
        Foods_sub_075_1.append(Food_set[i])
    elif df[Food_set[i]].mean() >=1 and df[Food_set[i]].mean()<2:
        Foods_sub_1_2.append(Food_set[i])
    elif df[Food_set[i]].mean() >=2 :
        Foods_sub_2_inf.append(Food_set[i])

means_household = df[Household_set].mean()
means_household.min()
means_household.max()
means_foods.value_counts()

Household_sub_0_025 = []
Household_sub_025_05 = []
Household_sub_05_075 = []
Household_sub_075_1 = []
Household_sub_1_2 = []
Household_sub_2_inf = []
for i in range(len(Household_set)):
    if df[Household_set[i]].mean() >=0 and df[Household_set[i]].mean()<0.25:
        Household_sub_0_025.append(Household_set[i])
    elif df[Household_set[i]].mean() >=0.25 and df[Household_set[i]].mean()<0.50:
        Household_sub_025_05.append(Food_set[i])
    elif df[Household_set[i]].mean() >=0.50 and df[Household_set[i]].mean()<0.75:
        Household_sub_05_075.append(Food_set[i])
    elif df[Household_set[i]].mean() >=0.75 and df[Household_set[i]].mean()<1:
        Household_sub_075_1.append(Food_set[i])
    elif df[Household_set[i]].mean() >=1 and df[Household_set[i]].mean()<2:
        Household_sub_1_2.append(Household_set[i])
    elif df[Food_set[i]].mean() >=2 :
        Household_sub_2_inf.append(Household_set[i])
     



Hobbies_sub_sets = [Hobbies_sub_0_025, Hobbies_sub_025_05, Hobbies_sub_05_075, Hobbies_sub_075_1, Hobbies_sub_1_2, Hobbies_sub_2_inf]
Food_sub_sets = [Foods_sub_0_025, Foods_sub_025_05, Foods_sub_05_075, Foods_sub_075_1, Foods_sub_1_2, Foods_sub_2_inf]
Household_sub_sets = [Household_sub_0_025, Household_sub_025_05, Household_sub_05_075, Household_sub_075_1, Household_sub_1_2, Household_sub_2_inf]

#Grouping by all sub sets
import numpy as np
Hobbies_series = []
for group1 in Hobbies_sub_sets:
    Hobbies_series.append(df[group1].apply(sum, axis = 1))

Food_series = []
for group2 in Food_sub_sets:
    Food_series.append(df[group2].apply(sum, axis = 1))

Household_series = []
for group3 in Household_sub_sets:
    Household_series.append(df[group3].apply(sum, axis = 1))


type(Hobbies_series)
Final_series = Hobbies_series+ Food_series+ Household_series

#Creating new dataset
data = pd.DataFrame(data = Final_series)
Data = data.transpose()
columns_name = ['Hobbies_1', 'Hobbies_2', 'Hobbies_3', 'Hobbies_4', 'Hobbies_5', 'Hobbies_6', 'Food_1', 'Food_2', 'Food_3', 'Food_4', 'Food_5', 'Food_6', 'Household_1', 'Household_2', 'Household_3', 'Household_4', 'Household_5','Household_6']
Data.columns = columns_name
Data.head()
Data.tail()
Data.index

#Visualization (basic)
for i in range(len(Data.columns)):
    Data[Data.columns[i]].plot(legend = True, figsize = (100,10)).legend(loc = 2, bbox_to_anchor = (1.0, 1.0))


#Holtwinter filter
from statsmodels.tsa.filters.hp_filter import hpfilter
import matplotlib.pyplot as plt
cycle, trend = hpfilter(Data['Hobbies_1'], lamb = 129600)

for i in range(len(Data.columns)):
    cycle, trend = hpfilter(Data[Data.columns[i]], lamb = 129600)
    plt.plot(cycle, color = 'red')
    plt.plot(trend, color = 'blue')
    plt.show()

#Seasonal decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(Data['Hobbies_1'], model = 'additive')
result.plot()


for i in range(len(Data.columns)):
    result = seasonal_decompose(Data[Data.columns[i]], model = 'additive')
    result.plot()


#Moving average
Data['Hobbies_1'].rolling(window = 7).mean().plot()

for i in range(len(Data.columns)):
    Data[Data.columns[i]].rolling(window = 7).mean().plot(figsize = (12,5)).legend(loc = 2, bbox_to_anchor = (1.0,1.0))

for i in range(len(Data.columns)):
    Data[Data.columns[i]].plot(figsize = (20,10)).legend(loc = 2, bbox_to_anchor = (1.0,1.0))
    Data[Data.columns[i]].rolling(window = 7).mean().plot(figsize = (20,10)).legend(loc = 2, bbox_to_anchor = (1.0,1.0))
    
#Exponential Smoothing {Double exponential smoothing}
from statsmodels.tsa.holtwinters import ExponentialSmoothing
span = 7
alpha = 2/(1+span)
model = ExponentialSmoothing(Data['Hobbies_1'], trend = 'add', seasonal = 'add', seasonal_periods = 7)
fitted_values = model.fit().fittedvalues
Data['Hobbies_1'].plot()
fitted_values.plot()

fitted_values = []
for i in range(len(Data.columns)):
    model = ExponentialSmoothing(Data[Data.columns[i]], trend = 'add', seasonal = 'add', seasonal_periods = 7)
    fitted_values.append(model.fit().fittedvalues)
fitted_values[0].tail()
Data['Hobbies_1'].shift(-1)


#Evaluation of the model
import numpy as np
from sklearn.metrics import mean_squared_error
error = mean_squared_error(y_true = Data['Hobbies_1'], y_pred = fitted_values)
np.sqrt(error)
Data.describe().iloc[:,11]

for i in range(len(Data.columns)):
    error = mean_squared_error(y_true = Data[Data.columns[i]], y_pred = fitted_values[i])
    print(f'The error of {Data.columns[i]} is {np.sqrt(error)}')

Data['Food_6'].describe()

#Lag plot
from pandas.plotting import lag_plot
lag_plot(Data['Hobbies_1'])
lag_plot(Data['Hobbies_2'])
lag_plot(Data['Hobbies_3'])
lag_plot(Data['Hobbies_4'])
lag_plot(Data['Hobbies_5'])
lag_plot(Data['Hobbies_6'])
lag_plot(Data['Food_1'])
lag_plot(Data['Food_2'])
lag_plot(Data['Food_3'])
lag_plot(Data['Food_4'])
lag_plot(Data['Food_5'])
lag_plot(Data['Food_6'])
lag_plot(Data['Household_1'])
lag_plot(Data['Household_2'])
lag_plot(Data['Household_3'])
lag_plot(Data['Household_4'])
lag_plot(Data['Household_5'])
lag_plot(Data['Household_6'])

#Auto correlation plot and partial cuto correlation plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(Data['Hobbies_1'], lags = 40)
plot_acf(Data['Hobbies_2'], lags = 40)
plot_acf(Data['Food_1'], lags = 40)
plot_acf(Data['Food_2'], lags = 40)

for i in range(len(Data.columns[12:18])):
    plot_acf(Data[Data.columns[i]], lags = 60)
Data.plot() 
    
#Auto Regression  series
import warnings
warnings.filterwarnings('ignore')
train_set = Data.iloc[0:1885]
test_set = Data.iloc[1885:]
from statsmodels.tsa.ar_model import AR,ARResults
import numpy as np
from sklearn.metrics import mean_squared_error
model = AR(Data['Food_5'])
Arfit = model.fit(ic = 'aic')  
Arfit.params    

predicted_Hobbies = []
for i in range(len(train_set.columns[0:6])):
    model_1 = AR(train_set[train_set.columns[i]])
    AR_23_fit_Hobbies = model_1.fit(maxlag = 23, method = 'cmle')
    predicted_Hobbies.append(AR_23_fit_Hobbies.predict(start = len(train_set), end = (len(Data)-1)))

for i in range(len(train_set.columns[0:6])):
    error1 = mean_squared_error(y_true = test_set[test_set.columns[i]], y_pred = predicted_Hobbies[i])
    print(f'The error of {train_set.columns[i]} is {np.sqrt(error1)}')

predicted_Foods = []
for i in train_set.columns[6:12]:
    model_2 = AR(train_set[i])
    AR_25_fit_Foods = model_2.fit(maxlag = 25, method = 'cmle')
    predicted_Foods.append(AR_25_fit_Foods.predict(start = len(train_set), end = (len(Data)-1)))

for i in range(len(train_set.columns[0:6])):
    error2 = mean_squared_error(y_true = test_set[test_set.columns[(i+6)]], y_pred = predicted_Foods[i])
    print(f'The error of {test_set.columns[(i+6)]} is {np.sqrt(error2)}')

predicted_Households = []
for i in train_set.columns[12:18]:
    model_3 = AR(train_set[i])
    AR_21_fit_Households = model_3.fit(maxlag = 21, method = 'cmle')
    predicted_Households.append(AR_21_fit_Households.predict(start = len(train_set), end = (len(Data)-1)))

for i in range(len(train_set.columns[0:6])):
    error3 = mean_squared_error(y_true = test_set[test_set.columns[(i+12)]], y_pred = predicted_Households[i])
    print(f'The error of {test_set.columns[(i+12)]} is {np.sqrt(error3)}')


'''
#Evaluation
from sklearn.metrics import mean_squared_error
import numpy as np
for i in range(len(train_set.columns)):
    error = mean_squared_error(y_true = test_set[test_set.columns[i]], y_pred = fitted_values_test[i])
    print(f'The error of {train_set.columns[i]} is {np.sqrt(error)}')

test_set['Food_6'].describe()
  
test_set['Hobbies_3'].plot(figsize = (12,5), legend = True) 
fitted_values_test[2].plot(figsize = (12,5), legend = True) 
#Forecasting

Forecast_Hobbies = []
for i in range(len(Data.columns[0:6])):
    model_1 = AR(Data[Data.columns[i]])
    AR_22_fit_Hobbies = model_1.fit(maxlag = 22, method = 'cmle')
    Forecast_Hobbies.append(AR_22_fit_Hobbies.predict(start = len(Data), end = 1940))

Forecast_Foods = []
for i in Data.columns[6:12]:
    model_2 = AR(Data[i])
    AR_24_fit_Foods = model_2.fit(maxlag = 24, method = 'cmle')
    Forecast_Foods.append(AR_24_fit_Foods.predict(start = len(Data), end = 1940))

Forecast_Households = []
for i in train_set.columns[12:18]:
    model_3 = AR(Data[i])
    AR_21_fit_Households = model_3.fit(maxlag = 21, method = 'cmle')
    Forecast_Households.append(AR_21_fit_Households.predict(start = len(Data), end = 1940))
    '''


#Forecasting through Exponential smoothing
import warnings
warnings.filterwarnings('ignore')
train_set = Data.iloc[0:1885]
test_set = Data.iloc[1885:]
from statsmodels.tsa.holtwinters import ExponentialSmoothing

fitted_values_test = []
for i in range(len(Data.columns)):
    model = ExponentialSmoothing(train_set[train_set.columns[i]], trend = 'add', seasonal = 'add', seasonal_periods = 7)
    fitted_values_test.append(model.fit().forecast(28))  
    
from sklearn.metrics import mean_squared_error
import numpy as np
for i in range(len(train_set.columns)):
    error = mean_squared_error(y_true = test_set[test_set.columns[i]], y_pred = fitted_values_test[i])
    print(f'The error of {train_set.columns[i]} is {np.sqrt(error)}')
    
    
Fitted_values_data = []
for i in range(len(Data.columns)):
    model = ExponentialSmoothing(Data[Data.columns[i]], trend = 'add', seasonal = 'add', seasonal_periods = 7)
    Fitted_values_data.append(model.fit().forecast(28))  

    


#Separating the datas


for j in range(28):
    for i in range(len(Hobbies_sub_0_025)):
        df[Hobbies_sub_0_025[i]][1885+j] = ((df[Hobbies_sub_0_025[i]][1885+j]/Hobbies_series[0][1885+j])*Fitted_values_data[0][j])


for j in range(28):
    for i in range(len(Hobbies_sub_025_05)):
        df[Hobbies_sub_025_05[i]][1885+j] = ((df[Hobbies_sub_025_05[i]][1885+j]/Hobbies_series[1][1885+j])*Fitted_values_data[1][j])

for j in range(28):
    for i in range(len(Hobbies_sub_05_075)):
        df[Hobbies_sub_05_075[i]][1885+j] = ((df[Hobbies_sub_05_075[i]][1885+j]/Hobbies_series[2][1885+j])*Fitted_values_data[2][j])

for j in range(28):
    for i in range(len(Hobbies_sub_075_1)):
        df[Hobbies_sub_075_1[i]][1885+j] = ((df[Hobbies_sub_075_1[i]][1885+j]/Hobbies_series[3][1885+j])*Fitted_values_data[3][j])

for j in range(28):
    for i in range(len(Hobbies_sub_1_2)):
        df[Hobbies_sub_1_2[i]][1885+j] = ((df[Hobbies_sub_1_2[i]][1885+j]/Hobbies_series[4][1885+j])*Fitted_values_data[4][j])

for j in range(28):
    for i in range(len(Hobbies_sub_2_inf)):
        df[Hobbies_sub_2_inf[i]][1885+j] = ((df[Hobbies_sub_2_inf[i]][1885+j]/Hobbies_series[5][1885+j])*Fitted_values_data[5][j])

for j in range(28):
    for i in range(len(Foods_sub_0_025)):
        df[Foods_sub_0_025[i]][1885+j] = ((df[Foods_sub_0_025[i]][1885+j]/Food_series[0][1885+j])*Fitted_values_data[6][j])

for j in range(28):
    for i in range(len(Foods_sub_025_05)):
        df[Foods_sub_025_05[i]][1885+j] = ((df[Foods_sub_025_05[i]][1885+j]/Food_series[1][1885+j])*Fitted_values_data[7][j])

for j in range(28):
    for i in range(len(Foods_sub_05_075)):
        df[Foods_sub_05_075[i]][1885+j] = ((df[Foods_sub_05_075[i]][1885+j]/Food_series[2][1885+j])*Fitted_values_data[8][j])

for j in range(28):
    for i in range(len(Foods_sub_075_1)):
        df[Foods_sub_075_1[i]][1885+j] = ((df[Foods_sub_075_1[i]][1885+j]/Food_series[3][1885+j])*Fitted_values_data[9][j])

for j in range(28):
    for i in range(len(Foods_sub_1_2)):
        df[Foods_sub_1_2[i]][1885+j] = ((df[Foods_sub_1_2[i]][1885+j]/Food_series[4][1885+j])*Fitted_values_data[10][j])

for j in range(28):
    for i in range(len(Foods_sub_2_inf)):
        df[Foods_sub_2_inf[i]][1885+j] = ((df[Foods_sub_2_inf[i]][1885+j]/Food_series[5][1885+j])*Fitted_values_data[11][j])

for j in range(28):
    for i in range(len(Household_sub_0_025)):
        df[Household_sub_0_025[i]][1885+j] = ((df[Household_sub_0_025[i]][1885+j]/Household_series[0][1885+j])*Fitted_values_data[12][j])

for j in range(28):
    for i in range(len(Household_sub_025_05)):
        df[Household_sub_025_05[i]][1885+j] = ((df[Household_sub_025_05[i]][1885+j]/Household_series[1][1885+j])*Fitted_values_data[13][j])

for j in range(28):
    for i in range(len(Household_sub_05_075)):
        df[Household_sub_05_075[i]][1885+j] = ((df[Household_sub_05_075[i]][1885+j]/Household_series[2][1885+j])*Fitted_values_data[14][j])

for j in range(28):
    for i in range(len(Household_sub_075_1)):
        df[Household_sub_075_1[i]][1885+j] = ((df[Household_sub_075_1[i]][1885+j]/Household_series[3][1885+j])*Fitted_values_data[15][j])

for j in range(28):
    for i in range(len(Household_sub_1_2)):
        df[Household_sub_1_2[i]][1885+j] = ((df[Household_sub_1_2[i]][1885+j]/Household_series[4][1885+j])*Fitted_values_data[16][j])

for j in range(28):
    for i in range(len(Household_sub_2_inf)):
        df[Household_sub_2_inf[i]][1885+j] = ((df[Household_sub_2_inf[i]][1885+j]/Household_series[5][1885+j])*Fitted_values_data[17][j])

Predicted_data = df.iloc[1885:,:]

Predicted_data.transpose().to_excel('read.xlsx')
