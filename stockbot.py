from pydataset import data
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import yfinance as yf
ticker = 'MSFT'

ticker = yf.Ticker(ticker) # initilize ticker - microsoft
msft_history = ticker.history(period='5mo') # get stock history of 5 motnh
df = pd.DataFrame(msft_history)
df = df.drop(['Open', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], axis=1).rename(columns={'High': 'Price'})
df['MA5'] = df.rolling(window=5)['Price'].mean()
df['MA20'] = df.rolling(window=20)['Price'].mean()
df['MADF'] = df['MA5'] - df['MA20']

dfx = df[['MA5', 'MA20', 'MADF']]
dfy = df['Price'].shift(-1)


print(df)