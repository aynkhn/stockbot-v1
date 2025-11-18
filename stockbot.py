from pydataset import data
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import yfinance as yf
import sys
from datetime import datetime
today = datetime.today().strftime('%Y-%m-%d')
ticker = 'MSFT' # interchangable

def df_actions(df):
    df = df[['High']].copy() # drop everything but high values
    df['MA5'] = df['High'].rolling(window=5).mean() # MA's
    df['MA20'] = df['High'].rolling(window=20).mean()
    df['MADF'] = df['MA5'] - df['MA20'] # MA difference
    df['Target'] = df['High'].shift(-1)
    return df

def cost_function(w, b, x, y):
    m = len(y)
    print(m)
    sigma = 0
    for i in range(m):
        sigma += (y.iloc[i] - ((w * x.iloc[i]) + b)) ** 2
        print(sigma)
    cost = sigma / (2 * m)
    return cost

def partial_derivatives(w, b, x, y):
    m = len(y)
    dw_sum = 0.0
    db_sum = 0.0
    error = 0
    for i in range(m):
            error = (y.iloc[i] - ((w * x.iloc[i]) + b))
            dw_sum += error * x.iloc[i]   # derivatlive w.r.t w
            db_sum += error          # derivative w.r.t b
    dw = -dw_sum / m
    db = -db_sum / m
    return dw, db

def gradient_descent(w, b):
    dw, db = partial_derivatives(w, b, x, y)
    tmp_w = w - (alpha * dw)
    tmp_b = b - (alpha * db)
    w = tmp_w
    b = tmp_b       
    return w, b

## TRAINING DATA ##
ticker = yf.Ticker(ticker) # initilize ticker
train_df = ticker.history(start="2020-01-01", end="2024-12-31") # full DF of stock history
train_df = df_actions(train_df) # returns df as: high,ma5,ma20,madf,target
train_df = train_df.dropna() # drop NaN's (lack of data)
x_train = train_df['MADF'] # MADF
y_train = train_df['Target'] # Tomorrow's Price

## TODAY'S DATA ##
today_df = ticker.history(period="1mo")
today_df = df_actions(today_df)
latest_row = today_df.iloc[-1]
today_madf = latest_row['MADF']

## REGRESSION ## 
w = 0
b = 0
prev_cost = float('inf')
x = x_train
y = y_train
alpha = 0.01
epsilon = 1e-6
max_iters = 2000

for i in range(max_iters):
    cost = cost_function(w, b, x, y)
    w, b = gradient_descent(w, b)
    if abs(prev_cost - cost) < epsilon:
        print(f"Stopped in {i} iterations")
        break
    else:
         prev_cost = cost
         continue
    
prediction = (w * today_madf) + b
print(prediction)