import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# imports.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
import yfinance as yf
from scipy.stats import ttest_ind
import torch
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates
import optuna
import functools
import pandas as pd

from imports import *
from constants import *

import matplotlib.dates as mdates

def plot_predictions_with_intervals(y_pred, lower_bound_res, upper_bound_res, dates, feature_names, std_dev=1):
    # Ensure we're only using the last 13 points for the forecast horizon
    forecast_length = FORECAST_RANGE
    y_pred = y_pred[-forecast_length:]
    lower_bound_res = lower_bound_res[-forecast_length:]
    upper_bound_res = upper_bound_res[-forecast_length:]
    dates = dates[-forecast_length:]

    assert len(dates) == forecast_length, f"Expected {forecast_length} dates, got {len(dates)}"

    n_features = y_pred.shape[1]
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 6*n_features), sharex=True)
    
    if n_features == 1:
        axes = [axes]
    
    z_score_percent = (stats.norm.cdf(std_dev) - stats.norm.cdf(-std_dev)) * 100
    
    for i, (ax, feature) in enumerate(zip(axes, feature_names)):
        y_pred_trimmed = y_pred[:, i]
        
        ax.plot(dates, y_pred_trimmed, label='Forecast', color='blue')
        ax.fill_between(dates, lower_bound_res[:, i], upper_bound_res[:, i], color='lightblue', alpha=0.4, label='Prediction Interval')
        
        ax.set_title(f'{feature} Forecast with Intervals\n(Based on residuals, ±{std_dev:.2f} std dev ≈ {z_score_percent:.2f}%)')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        
        ax.legend(loc='upper left')
        
        # Ensure x-axis shows exactly 13 dates
        ax.set_xticks(dates)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.gcf().autofmt_xdate()  # Rotate date labels
    plt.tight_layout()
    plt.savefig('forecast_intervals.png')