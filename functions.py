from imports import *



def translate_column_name(column_name):
    """
    Translates a column name from string format to tuple format.
    
    Args:
    column_name (str or tuple): Column name in string format, e.g., 'stock.GOOGL.Adj Close',
                                or already in tuple format, e.g., ('stock', 'GOOGL', 'Adj Close').
    
    Returns:
    tuple: Column name in tuple format, e.g., ('stock', 'GOOGL', 'Adj Close').
    """
    if isinstance(column_name, tuple):
        return column_name
    return tuple(column_name.split('.'))

def calculate_threshold(df_length, confidence_level=0.95):
    # Degrees of freedom
    df = df_length - 2
    
    # Calculate the critical value for the Pearson correlation coefficient
    critical_value = stats.t.ppf((1 + confidence_level) / 2., df)
    threshold = (critical_value**2 / (critical_value**2 + df))**0.5
    return threshold

def calculate_optimal_lags(forecast_feature, feature, window_size=91):
    """
    Calculate the optimal lag for a feature based on median rolling window correlation with the forecast feature.
    Also returns all significant correlations exceeding the given threshold.

    Parameters:
    - df (pd.DataFrame): The data frame containing the features.
    - forecast_feature (str): The name of the forecast feature.
    - feature (str): The feature for which to calculate the lag.
    - window_size (int): The size of the rolling window (default: 91 days).

    Returns:
    - significant_lags (list): List of lags with significant correlations, sorted by magnitude of correlation.
    """
    
    correlations = {}
    significant_correlations = []

    for lag in range(1, window_size + 1):
        shifted_feature = feature.shift(lag)
        rolling_corr = forecast_feature.rolling(window=window_size).corr(shifted_feature)
        rolling_corr = rolling_corr.dropna()
        threshold = calculate_threshold(len(rolling_corr))
        median_corr = rolling_corr.median()
        correlations[lag] = median_corr

        if abs(median_corr) >= threshold:
            significant_correlations.append((lag, median_corr))

    # Sort by the absolute value of the correlation in descending order
    significant_correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    # Extract the lags from the sorted significant correlations
    significant_lags = [lag for lag, _ in significant_correlations]

    #return highest
    return [significant_lags[0]]