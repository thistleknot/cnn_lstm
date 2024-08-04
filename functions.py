import pandas as pd

def calculate_optimal_lag(df, forecast_feature, feature, window_size=91, threshold=0.5):
    """
    Calculate the optimal lag for a feature based on median rolling window correlation with the forecast feature.
    Also returns all significant correlations exceeding the given threshold.

    Parameters:
    - df (pd.DataFrame): The data frame containing the features.
    - forecast_feature (str): The name of the forecast feature.
    - feature (str): The feature for which to calculate the lag.
    - window_size (int): The size of the rolling window (default: 91 days).
    - threshold (float): The correlation threshold for considering a correlation significant (default: 0.5).

    Returns:
    - optimal_lag (int): The lag with the highest median correlation.
    - significant_correlations (list): List of tuples containing (lag, median correlation) for significant correlations.
    """
    correlations = {}
    significant_correlations = []

    for lag in range(1, window_size + 1):
        shifted_feature = df[feature].shift(lag)
        rolling_corr = df[forecast_feature].rolling(window=window_size).corr(shifted_feature)
        median_corr = rolling_corr.median()
        correlations[lag] = median_corr

        if abs(median_corr) >= threshold:
            significant_correlations.append((lag, median_corr))

    optimal_lag = max(correlations, key=correlations.get)
    return optimal_lag, significant_correlations
