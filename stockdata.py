import pandas as pd
import yfinance as yf
import numpy as np

tickers = ["SPY","QQQ","IWM","XLF","XLK","XLE","XLV","TLT","GLD","HYG"]

data = yf.download(tickers, start="2009-01-01", end="2012-01-01", auto_adjust=True)["Close"]

log_ret = np.log(data / data.shift(1)).dropna()

log_ret.to_csv("C:/Users/H/Documents/projects/Avarent/data/etf_daily_log_returns_new_time_period2.csv")