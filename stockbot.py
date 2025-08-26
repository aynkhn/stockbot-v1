from pydataset import data
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import yfinance as yf
ticker = 'MSFT' # interchangable

def df_actions(df):
    df = df[['High']].copy() # drop everything but high values
    df['MA5'] = df['High'].rolling(window=5).mean() # MA's
    df['MA20'] = df['High'].rolling(window=20).mean()
    df['MADF'] = df['MA5'] - df['MA20'] # MA difference
    df['Target'] = df['High'].shift(-1)
    return df.dropna() # drop NaN

## TRAINING DATA ##
ticker = yf.Ticker(ticker) # initilize ticker
train_df = ticker.history(start="2020-01-01", end="2024-12-31") # get stock history
test_df = ticker.history(start="2025-01-01", end="2025-12-31")
train_df = df_actions(train_df)
test_df = df_actions(test_df)
x_train = train_df[['MA5', 'MA20', 'MADF']] # X train
y_train = train_df['Target'] # Y train
x_test = test_df[['MA5', 'MA20', 'MADF']] # X test
y_test = test_df['Target'] # Y test

## REGRESSION ## 
LR = LinearRegression() # blank regression LR
LR.fit(x_train, y_train) # fit LR to data

## TEST AND DISPLAY ##
# y_pred = LR.predict(x_test) # run test data
# plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
# lo = min(y_test.min(), y_pred.min())
# hi = max(y_test.max(), y_pred.max())
# plt.plot([lo, hi], [lo, hi], color='red', lw=2, label='Ideal: y = x')
# coeffs = np.polyfit(y_test, y_pred, 1)  # slope & intercept
# fit_line = np.poly1d(coeffs)
# plt.plot([lo, hi], fit_line([lo, hi]), color='green', lw=2, label='Best Fit')
# plt.xlabel("Actual Price")
# plt.ylabel("Predicted Price")
# plt.title("Regression Predictions")
# plt.legend()
# plt.show()
# print("Coefficients:", LR.coef_) # testing
# print("Intercept:", LR.intercept_)
# print("R² score:", LR.score(x_test, y_test))

latest = ticker.history(period='1mo')
df_latest = df_actions(latest)
today_features = df_latest[['MA5','MA20','MADF']].tail(1)
predicted_price = LR.predict(today_features)
print("Predicted High Price for tomorrow:", predicted_price[0])

