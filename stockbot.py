from pydataset import data
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import yfinance as yf
ticker = 'MSFT' # interchangable

ticker = yf.Ticker(ticker) # initilize ticker
msft_history = ticker.history(period='1y') # get stock history of 5 motnh
df = pd.DataFrame(msft_history)
df = df[['High']] # drop everything but high values
df['MA5'] = df['High'].rolling(window=5).mean() # MA's
df['MA20'] = df['High'].rolling(window=20).mean()
df['MADF'] = df['MA5'] - df['MA20'] # MA difference
df['Target'] = df['High'].shift(-1)
df = df.dropna() # drop NaN
dfx = df[['MA5', 'MA20', 'MADF']] # X
dfy = df['Target'] # Y

x_train, x_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.25, random_state=42) # split train and test data, seeded

LR = LinearRegression() # blank regression LR
LR.fit(x_train, y_train) # fit LR to data

y_pred = LR.predict(x_test) # run test data

# display code
plt.scatter(y_test, y_pred, color='blue', label='Test Data')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, label='Regression')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear Regression Predictions")
plt.legend()
plt.show()

print("Coefficients:", LR.coef_) # testing
print("Intercept:", LR.intercept_)
print("R² score:", LR.score(x_test, y_test))
