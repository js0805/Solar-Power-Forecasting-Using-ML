# -*- coding: utf-8 -*-
"""
Created on Wed May 27 22:39:56 2020

@author: js0805
"""
# Importing required libraries 
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose 

df = pd.read_excel("F:/ml/minor/Data.xlsx")

df = df.sort_values('Date')


df.isnull().sum()

df = df.groupby('Date')['Energy'].sum().reset_index()

df = df.set_index('Date')
df.index

y = df['Energy'].resample('MS').mean()


y.plot(figsize=(15, 6))
plt.show()

# Print the first five rows of the dataset 
df.head() 

# ETS Decomposition 
result = seasonal_decompose(df['Energy'], 
							model ='multiplicative') 

# ETS plot 
result.plot()


# Import the library 
from pmdarima import auto_arima 
  
# Ignore harmless warnings 
import warnings 
warnings.filterwarnings("ignore") 
  
# Fit auto_arima function to AirPassengers dataset 
stepwise_fit = auto_arima(df['Energy'], start_p = 1, start_q = 1, 
                          max_p = 3, max_q = 3, m = 12, 
                          start_P = 0, seasonal = True, 
                          d = None, D = 1, trace = True, 
                          error_action ='ignore',   # we don't want to know if an order does not work 
                          suppress_warnings = True,  # we don't want convergence warnings 
                          stepwise = True)           # set to stepwise 
  
# To print the summary 
stepwise_fit.summary()  

# Split data into train / test sets 
train = df.iloc[:len(df)-12] 
test = df.iloc[len(df)-12:] # set one year(12 months) for testing 

# Fit a SARIMAX(2, 0, 3)x(0, 1, [1, 2], 12) on the training set 
from statsmodels.tsa.statespace.sarimax import SARIMAX 

model = SARIMAX(train['Energy'], 
				order = (2, 0, 3), 
				seasonal_order =(0, 1, [1, 2], 12)) 

result = model.fit() 
result.summary() 


start = len(train) 
end = len(train) + len(test) - 1

# Predictions for one-year against the test set 
predictions = result.predict(start, end, 
							typ = 'levels').rename("Predictions") 

# plot predictions and actual values 
predictions.plot(legend = True) 
test['Energy'].plot(legend = True) 

# Load specific evaluation tools 
from sklearn.metrics import mean_squared_error 
from statsmodels.tools.eval_measures import rmse 

# Calculate root mean squared error 
rmse(test["Energy"], predictions) 

# Calculate mean squared error 
mean_squared_error(test["Energy"], predictions) 

# Train the model on the full dataset 
model = model = SARIMAX(df['Energy'], 
						order = (2, 0, 3), 
						seasonal_order =(0, 1, [1, 2], 12)) 
result = model.fit() 

# Forecast for the next 3 years 
forecast = result.predict(start = len(df), 
						end = (len(df)-1) + 3 * 30, 
						typ = 'levels').rename('Forecast') 

# Plot the forecast values 
df['Energy'].plot(figsize = (12, 5), legend = True,alpha = 0.7,linewidth =2) 
forecast.plot(legend = True,linewidth = 2) 

