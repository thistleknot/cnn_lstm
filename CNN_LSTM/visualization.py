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

def plot_predictions_with_intervals(y_pred, lower_bound_res, upper_bound_res, dates, feature_names, std_dev=0.67):
    n_features = y_pred.shape[1]
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 6*n_features), sharex=True)
    
    if n_features == 1:
        axes = [axes]
    
    # Calculate the percentage for ±0.67 standard deviations
    z_score_percent = (stats.norm.cdf(std_dev) - stats.norm.cdf(-std_dev)) * 100
    
    for i, (ax, feature) in enumerate(zip(axes, feature_names)):
        y_pred_trimmed = y_pred[:, i]
        
        # Plot predictions
        ax.plot(dates, y_pred_trimmed, label='Predicted', color='red')
        
        # Plot residual-based prediction intervals
        ax.fill_between(dates, lower_bound_res[:, i], upper_bound_res[:, i], color='green', alpha=0.2, label='Residual-Based Prediction Interval')
        
        # Add title including the z-score percentage
        ax.set_title(f'{feature} Prediction with Intervals\n(Based on residuals, ±0.67 std dev ≈ {z_score_percent:.2f}%)')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        
        ax.legend(loc='upper left')
        
        # Format x-axis to show all dates
        ax.set_xticks(dates)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))

    plt.gcf().autofmt_xdate()  # Rotate date labels
    plt.tight_layout()
    plt.savefig('predictions_intervals.png')

