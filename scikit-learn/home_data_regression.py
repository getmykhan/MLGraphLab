# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 17:13:22 2017

@author: Mohammed Yusuf Khan
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import statsmodels.formula.api as sms
import seaborn as sns


df = pd.read_csv('home_data.csv')

df.describe()['price'].round(3)


price_col = df['price']

df.drop('price', axis = 1, inplace = True)

df['price'] = price_col

df.drop('date', axis = 1, inplace = True)


""" Analysis by droping the date column """

X = df.iloc[:,:-1].values
y = df.iloc[:, -1:].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

regressor = LinearRegression()

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)


""" Backward Eliminations to make it more accurate """
## Significance level set to 0.5

X = np.append(arr = np.ones((len(X), 1)).astype(int), values= X, axis = 1)



x_opt = X[:,0:]
regressor_OLS = sms.OLS(endog=y, exog=x_opt).fit()
print(regressor_OLS.summary())

#sns.distplot(y_pred)
sns.jointplot(y_test, y_pred, kind='reg', size =10,scatter_kws={"s": 1})

#plt.scatter(y_test, regressor.predict(X_test), s= 1)
#plt.plot(X_train, regressor.predict(X_train),color= 'red')
#plt.plot(X_train[:,1], regressor.predict(X_train[:,1]))
#plt.show()

#plt.scatter(X_train[:,0], y_train, color = 'blue')
#plt.plot(X_train[:,0], regressor.predict(X_train[:,0]),color = 'red')
#plt.show()