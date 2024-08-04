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

def plot_predictions_with_intervals(y_pred, lower_bound_res, upper_bound_res, dates, feature_names, std_dev=0.33):
    n_features = y_pred.shape[1]
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 6*n_features), sharex=True)
    
    if n_features == 1:
        axes = [axes]
    
    for i, (ax, feature) in enumerate(zip(axes, feature_names)):
        y_pred_trimmed = y_pred[:, i]
        
        # Plot predictions
        ax.plot(dates, y_pred_trimmed, label='Predicted', color='red')
        
        # Plot residual-based prediction intervals
        ax.fill_between(dates, lower_bound_res[:, i], upper_bound_res[:, i], color='green', alpha=0.2, label='Residual-Based Prediction Interval')
        
        # Calculate the percentage of the standard deviation
        mean = y_pred_trimmed.mean()
        interval = std_dev * mean
        percentage = std_dev * 100
        
        # Add title including the prediction interval percentage
        ax.set_title(f'{feature} Prediction with Intervals (±{percentage:.2f}% std dev)')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        
        ax.legend(loc='upper left')
        
        # Format x-axis to show dates nicely
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.gcf().autofmt_xdate()  # Rotate date labels
    plt.tight_layout()
    plt.savefig('predictions_intervals.png')

