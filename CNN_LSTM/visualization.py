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

def plot_predictions_with_intervals_and_returns(y_pred, lower_bound_res, upper_bound_res, dates, feature_names, last_actual_price):
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
        ax_price.set_title(f'{feature} Close Forecast with 95% Prediction Interval')
        ax_price.set_xlabel('Date')
        ax_price.set_ylabel('Price')
        ax_price.legend(loc='upper left')
        
        # Plot 2: Cumulative Returns and Second Derivative
        # Calculate cumulative returns based on the last actual price
        cumulative_returns = (y_pred_trimmed - last_actual_price) / last_actual_price
        
        # Calculate the second derivative
        second_derivative = pd.Series(cumulative_returns).diff().diff()
        
        # Plot cumulative returns
        ax_return.plot(dates, cumulative_returns, label='Cumulative Return', color='green')
        ax_return.set_ylabel('Cumulative Return', color='green')
        ax_return.tick_params(axis='y', labelcolor='green')
        
        # Create a twin axis for the second derivative
        ax_second_derivative = ax_return.twinx()
        ax_second_derivative.set_ylabel('Second Derivative', color='red')
        ax_second_derivative.tick_params(axis='y', labelcolor='red')
        
        # Find the inflection point (where second derivative becomes negative)
        inflection_points = second_derivative[second_derivative < 0].index
        if len(inflection_points) > 0:
            inflection_point = inflection_points[0]
            inflection_date = dates[inflection_point]
            ax_return.axvline(x=inflection_date, color='red', linestyle='--', label='Inflection Point')
            
            # Adjust the y-axis of the second derivative to center 0 at the inflection point
            max_abs_value = max(abs(second_derivative.min()), abs(second_derivative.max()))
            ax_second_derivative.set_ylim(-max_abs_value, max_abs_value)
            
            # Add a horizontal line at y=0 for the second derivative
            ax_second_derivative.axhline(y=0, color='gray', linestyle=':', alpha=0.7)
        
        ax_return.set_title(f'{feature} Cumulative Return and Second Derivative')
        ax_return.set_xlabel('Date')
        
        # Combine legends
        lines_return, labels_return = ax_return.get_legend_handles_labels()
        ax_return.legend(lines_return, labels_return, loc='upper left')
        
        # Format x-axis for both plots
        for ax in [ax_price, ax_return]:
            ax.set_xticks(dates)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    plt.gcf().autofmt_xdate()  # Rotate date labels
    plt.tight_layout()
    plt.savefig('forecast_intervals_and_returns.png')
    plt.close(fig)  # Close the figure to free up memory