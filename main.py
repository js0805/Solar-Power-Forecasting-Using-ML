import pandas as pd


dataset= pd.read_excel('dataset.xlsx',sheet_name='Sheet1')
dataset.drop({'Date'} , axis =1,inplace=True)
X = dataset.iloc[:,0:5].values
y = dataset.iloc[:,5:6].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)