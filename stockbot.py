from pydataset import data
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import yfinance as yf
 
today = pd.Timestamp.today(tz="America/New_York").normalize()
past_20 = today - pd.Timedelta(days=20)
past_5 = today - pd.Timedelta(days=5)
 
msft = yf.Ticker("MSFT")
msft_history = msft.history(period="1mo")
msft_high_20 = msft_history.loc[past_20:today, "High"].tolist()
 
print(msft_high_20)