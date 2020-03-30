# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 23:26:47 2020

@author: Js0805
"""
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

 
dataset_1= pd.read_excel('dataset.xlsx',sheet_name='Sheet1')
cols =['AT','WS','RH','BP','SR']
dataset_1.drop(cols, axis=1, inplace=True)

dataset_1 = dataset_1.sort_values('Date')

dataset_1.isnull().sum()

dataset_1 = dataset_1.groupby('Date')['Energy (Meters)'].sum().reset_index()

dataset_1 = dataset_1.set_index('Date')
dataset_1.index
print(dataset_1.head())
dataset_1.plot()
plt.show()
autocorrelation_plot(dataset_1)
model = ARIMA(dataset_1, order=(10,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

# plot residual errors
from pandas import DataFrame
residuals = DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())

X = dataset_1.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()