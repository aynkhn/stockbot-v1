from pydataset import data
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import yfinance as yf
ticker = 'MSFT'

ticker = yf.Ticker(ticker) # initilize ticker - microsoft
msft_history = ticker.history(period='1y') # get stock history of 5 motnh
df = pd.DataFrame(msft_history)
df = df[['High']] # drop everything but high values
df['MA5'] = df.rolling(window=5)['High'].mean() # MA's
df['MA20'] = df.rolling(window=20)['High'].mean()
df['MADF'] = df['MA5'] - df['MA20'] # MA difference
df['Target'] = df['High'].shift(-1)
df = df.dropna() # drop NaN
dfx = df[['MA5', 'MA20', 'MADF']] # X
dfy = df['Target'] # Y

x_train, x_test, y_train, y_test = train_test_split(dfx, dfy) # split train and test data

LR = LinearRegression() # blank regression LR
LR.fit(x_train, y_train) # fit LR to data

y_pred = LR.predict(x_test) # run test data

# display code
plt.scatter(y_test, y_pred, color='blue', label='Test Data')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, label='Regression')
plt.xlabel("Actual High")
plt.ylabel("Predicted High")
plt.title("Linear Regression Predictions")
plt.legend()
plt.show()