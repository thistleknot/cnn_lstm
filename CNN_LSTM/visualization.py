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

def plot_predictions_with_intervals(y_pred, lower_bound_res, upper_bound_res, dates, feature_names):
    forecast_length = len(dates)
    y_pred = y_pred[-forecast_length:]
    lower_bound_res = lower_bound_res[-forecast_length:]
    upper_bound_res = upper_bound_res[-forecast_length:]
    dates = dates[-forecast_length:]

    assert len(dates) == forecast_length, f"Expected {forecast_length} dates, got {len(dates)}"

    n_features = y_pred.shape[1]
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 6*n_features), sharex=True)
    
    if n_features == 1:
        axes = [axes]
    
    for i, (ax, feature) in enumerate(zip(axes, feature_names)):
        y_pred_trimmed = y_pred[:, i]
        
        # Plot the forecast line
        ax.plot(dates, y_pred_trimmed, label='Forecast', color='blue')
        
        # Plot 95% prediction intervals
        ax.fill_between(dates, lower_bound_res[:, i], upper_bound_res[:, i], color='lightblue', alpha=0.4, label='95% Prediction Interval')
        
        ax.set_title(f'{feature} Close Forecast with 95% Prediction Interval')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend(loc='upper left')
        
        # Set x-axis ticks and format
        ax.set_xticks(dates)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.gcf().autofmt_xdate()  # Rotate date labels
    plt.tight_layout()
    plt.savefig('forecast_intervals.png')
    plt.close(fig)  # Close the figure to free up memory