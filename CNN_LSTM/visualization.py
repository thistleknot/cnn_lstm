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

def plot_predictions_with_intervals_and_returns(y_pred, lower_bound_res, upper_bound_res, dates, feature_names):
    forecast_length = len(dates)
    y_pred = y_pred[-forecast_length:]
    lower_bound_res = lower_bound_res[-forecast_length:]
    upper_bound_res = upper_bound_res[-forecast_length:]
    dates = dates[-forecast_length:]

    assert len(dates) == forecast_length, f"Expected {forecast_length} dates, got {len(dates)}"

    n_features = y_pred.shape[1]
    fig, axes = plt.subplots(n_features, 2, figsize=(20, 8*n_features), sharex='col')
    
    if n_features == 1:
        axes = axes.reshape(1, -1)
    
    for i, (ax_price, ax_return, feature) in enumerate(zip(axes[:, 0], axes[:, 1], feature_names)):
        y_pred_trimmed = y_pred[:, i]
        
        # Plot 1: Price Forecast with Intervals
        ax_price.plot(dates, y_pred_trimmed, label='Forecast', color='blue')
        ax_price.fill_between(dates, lower_bound_res[:, i], upper_bound_res[:, i], color='lightblue', alpha=0.4, label='95% Prediction Interval')
        
        # Calculate rate of change
        roc = pd.Series(y_pred_trimmed).pct_change()
        # Calculate moving average of rate of change (you can adjust the window size)
        roc_ma = roc.rolling(window=5).mean()
        
        # Find crossover points
        crossover = np.where((roc.shift(1) > roc_ma.shift(1)) & (roc <= roc_ma))[0]
        
        # Draw vertical lines at crossover points
        for idx in crossover:
            if idx < len(dates):
                ax_price.axvline(x=dates[idx], color='red', linestyle='--', alpha=0.7)
                ax_return.axvline(x=dates[idx], color='red', linestyle='--', alpha=0.7)
        
        ax_price.set_title(f'{feature} Close Forecast with 95% Prediction Interval')
        ax_price.set_xlabel('Date')
        ax_price.set_ylabel('Price')
        ax_price.legend(loc='upper left')
        
        # Plot 2: Cumulative Returns
        returns = pd.Series(y_pred_trimmed).pct_change()
        cumulative_returns = (1 + returns).cumprod() - 1
        ax_return.plot(dates, cumulative_returns, label='Cumulative Return', color='green')
        
        ax_return.set_title(f'{feature} Cumulative Return')
        ax_return.set_xlabel('Date')
        ax_return.set_ylabel('Cumulative Return')
        ax_return.legend(loc='upper left')
        
        # Format x-axis for both plots
        for ax in [ax_price, ax_return]:
            ax.set_xticks(dates)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.gcf().autofmt_xdate()  # Rotate date labels
    plt.tight_layout()
    plt.savefig('forecast_intervals_and_returns.png')
    plt.close(fig)  # Close the figure to free up memory