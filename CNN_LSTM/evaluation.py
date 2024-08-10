from .training import train_model
from .CNN_TS import LSTMEncoderDecoderAttention
from imports import *
from constants import *
from .visualization import plot_predictions_with_intervals_and_returns

def translate_column_name(column_name):
    # Split the string into its components
    components = column_name.split('.')
    # Return as a tuple matching the DataFrame's multi-level index structure
    return tuple(components)

# Function to split sequence for LSTM
def split_sequence(sequence, look_back, forecast_range, forecast_indices):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + look_back
        out_end_ix = end_ix + forecast_range
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix, forecast_indices]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def calculate_prediction_intervals_based_on_residuals(yhat_forecast_inverse, std_devs_time_step, confidence_level=0.95):
    # Determine the z-score for the specified confidence level (95%)
    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    
    # Calculate the prediction intervals
    lower_bounds_residuals = yhat_forecast_inverse - z_score * std_devs_time_step
    upper_bounds_residuals = yhat_forecast_inverse + z_score * std_devs_time_step
    
    return lower_bounds_residuals, upper_bounds_residuals

def inverse_transform(scaler, y_test, yhat, forecast_indices):
    y_test_reshaped = y_test.reshape(-1, y_test.shape[-1])
    yhat_reshaped = yhat.reshape(-1, yhat.shape[-1])
    
    y_test_full = np.zeros((y_test_reshaped.shape[0], len(scaler.scale_)))
    yhat_full = np.zeros((yhat_reshaped.shape[0], len(scaler.scale_)))
    
    y_test_full[:, forecast_indices] = y_test_reshaped
    yhat_full[:, forecast_indices] = yhat_reshaped
    
    y_test_inverse = scaler.inverse_transform(y_test_full)
    yhat_inverse = scaler.inverse_transform(yhat_full)
    
    y_test_inverse = y_test_inverse[:, forecast_indices]
    yhat_inverse = yhat_inverse[:, forecast_indices]
    
    return yhat_inverse, y_test_inverse

def evaluate_forecast(y_test_inverse, yhat_inverse):
    mse = np.mean((y_test_inverse - yhat_inverse) ** 2)
    mae = np.mean(np.abs(y_test_inverse - yhat_inverse))
    mape = np.mean(np.abs((y_test_inverse - yhat_inverse) / y_test_inverse)) * 100
    return mae, mse, mape

def calculate_per_residual_intervals(residuals_per_simulation, std_dev_constant=1):
    """
    Calculate per-residual confidence intervals based on the standard deviation
    of residuals at each forecast point across all simulations.

    Parameters:
    - residuals_per_simulation (np.ndarray): Residuals for each forecast point 
      across all simulations, with shape (num_simulations, forecast_range, num_features).
    - std_dev_constant (float): The number of standard deviations for the interval.

    Returns:
    - lower_bounds (np.ndarray): Lower bounds of the prediction intervals.
    - upper_bounds (np.ndarray): Upper bounds of the prediction intervals.
    """
    std_devs_per_point = np.std(residuals_per_simulation, axis=0)
    mean_forecast = np.mean(residuals_per_simulation, axis=0)  # Calculate mean forecast
    lower_bounds = mean_forecast - std_dev_constant * std_devs_per_point
    upper_bounds = mean_forecast + std_dev_constant * std_devs_per_point

    return lower_bounds, upper_bounds

def objective(trial, combined_df, forecast_features, trial_results, include_flags, LOOK_BACK, FORECAST_RANGE, NUM_EPOCHS):
    # Select features to include using trial suggestions
    selected_flags = [trial.suggest_categorical(flag, [True, False]) for flag in include_flags]
    
    # Construct the feature list based on the trial's suggestions
    selected_features = [features[0]]  # Always include the main feature
    for index, flag in enumerate(selected_flags, 1):
        if flag:
            selected_features.append(features[index])
    
    # Convert selected_features to tuples to match combined_df's multi-level column index
    selected_features = [tuple(feature.split('.')) for feature in selected_features]
    
    # Ensure that the selected features exist in the combined_df
    valid_features = [feature for feature in selected_features if feature in combined_df.columns]
    
    if(debug):
        # Debug print statement to verify selected and valid features
        print(f"Selected features: {selected_features}")
        print(f"Valid features: {valid_features}")
        print(f"Combined_df columns: {combined_df.columns.tolist()}")

    if not valid_features:
        raise ValueError("No valid features selected for training.")
    
    # Extract and process the valid features
    data_subset = combined_df[valid_features].dropna()
    
    # Handle the case where the data subset is empty after dropping NaNs
    if data_subset.empty:
        raise ValueError("No data available after dropping NaNs.")
    
    # Scaling the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_subset)
    scaled_data = pd.DataFrame(scaled_data, columns=data_subset.columns, index=data_subset.index)
    
    # Flatten the multi-index columns for easier mapping
    scaled_data.columns = ['.'.join(col) for col in scaled_data.columns]
    
    # Derive feature mapping from combined_df
    feature_mapping = {col: idx for idx, col in enumerate(scaled_data.columns)}
    
    # Map forecast features to their indices
    forecast_indices = [feature_mapping[feature] for feature in forecast_features]
    n_outputs = len(forecast_indices)
    
    mape_results = []
    for _ in range(NUM_SIMULATIONS):
        # Prepare the data for LSTM
        X, y = split_sequence(scaled_data.values, LOOK_BACK, FORECAST_RANGE, forecast_indices)
        
        if X.size == 0 or y.size == 0:
            return float('inf')  # Return an invalid score if there's no data
        
        X_train = torch.tensor(X, dtype=torch.float32).to(device)
        y_train = torch.tensor(y, dtype=torch.float32).to(device)
        
        n_features = X_train.shape[2] if X_train.ndimension() > 2 else 0
        
        # Define the model
        hidden_dim = 24
        
        model = LSTMEncoderDecoderAttention(LOOK_BACK, FORECAST_RANGE, n_features, hidden_dim, n_outputs).to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        batch_size = 32
        early_stopping_patience = 10
        
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        
        train_model(model, train_loader, criterion, optimizer, NUM_EPOCHS, early_stopping_patience)
        
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        
        model.eval()
        with torch.no_grad():
            yhat = model(X_train).cpu().numpy()
        
        yhat_inverse, y_train_inverse = inverse_transform(scaler, y_train.cpu().numpy(), yhat, forecast_indices)
        
        mae, mse, mape = evaluate_forecast(y_train_inverse, yhat_inverse)
        mape_results.append(mape)
    
    average_mape = np.mean(mape_results)
    
    # Store the results
    trial_results.append({
        'trial_number': trial.number,
        'mape': average_mape,
        **{include_flags[i-1]: selected_flags[i-1] for i in range(1, len(features))}
    })
    
    return average_mape

def backtest_and_evaluate(study, combined_df, scaler, features, forecast_features, LOOK_BACK=156, FORECAST_RANGE=13, NUM_EPOCHS=20):
    # Convert features and forecast_features to tuples to match combined_df's multi-level column index
    features = [tuple(feature.split('.')) for feature in features]
    forecast_features = [tuple(feature.split('.')) for feature in forecast_features]
    
    # Select valid features from combined_df
    selected_columns = [col for col in combined_df.columns if col in features]
    
    if(debug):
    # Debug print statements to check selected columns and data_subset
        print(f"Features: {features}")
        print(f"Selected columns: {selected_columns}")
    
    data_subset = combined_df[selected_columns].dropna()
    
    # Debug print statement to check data_subset
    if(debug):
        print(f"Data subset shape: {data_subset.shape}")
        print(f"Data subset head:\n{data_subset.head()}")
    
    if data_subset.empty:
        raise ValueError("No data available after dropping NaNs.")
    
    scaled_data = scaler.fit_transform(data_subset)
    scaled_data = pd.DataFrame(scaled_data, columns=data_subset.columns, index=data_subset.index)
    
    # The rest of your function remains unchanged
    scaled_data.columns = ['.'.join(col) if isinstance(col, tuple) else col for col in scaled_data.columns]
    feature_mapping = {col: idx for idx, col in enumerate(scaled_data.columns)}
    
    forecast_features = ['.'.join(feature) if isinstance(feature, tuple) else feature for feature in forecast_features]
    forecast_indices = [feature_mapping[feature] for feature in forecast_features]
    
    X, y = split_sequence(scaled_data.values, LOOK_BACK, FORECAST_RANGE, forecast_indices)
    
    if X.size == 0 or y.size == 0:
        return
    
    X_train = torch.tensor(X, dtype=torch.float32).to(device)
    y_train = torch.tensor(y, dtype=torch.float32).to(device)
    
    hidden_dim = 24
    n_features = X_train.shape[2] if X_train.ndimension() > 2 else 0
    
    model = LSTMEncoderDecoderAttention(LOOK_BACK, FORECAST_RANGE, n_features, hidden_dim, len(forecast_indices)).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    batch_size = 32
    early_stopping_patience = 10
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    
    train_model(model, train_loader, criterion, optimizer, NUM_EPOCHS, early_stopping_patience)
    
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    
    model.eval()
    with torch.no_grad():
        yhat = model(X_train).cpu().numpy()
    
    yhat_inverse, y_train_inverse = inverse_transform(scaler, y_train.cpu().numpy(), yhat, forecast_indices)
    
    y_train_inverse_time_step = y_train_inverse.reshape(-1, FORECAST_RANGE, y_train_inverse.shape[-1])
    yhat_inverse_time_step = yhat_inverse.reshape(-1, FORECAST_RANGE, yhat_inverse.shape[-1])
    
    # Compute residuals and standard deviations across both time steps and features
    residuals_time_step = y_train_inverse_time_step - yhat_inverse_time_step
    std_devs_time_step = np.std(residuals_time_step, axis=0)  # Shape: (FORECAST_RANGE, n_features)
    
    last_values = scaled_data.values[-LOOK_BACK:]
    X_last = np.expand_dims(last_values, axis=0)
    X_last = torch.tensor(X_last, dtype=torch.float32).to(device)
    
    model.eval()
    with torch.no_grad():
        yhat_forecast = model(X_last).cpu().numpy()
    
    yhat_forecast = yhat_forecast.reshape(-1, len(forecast_indices))
    yhat_forecast_inverse, _ = inverse_transform(scaler, np.zeros_like(yhat_forecast), yhat_forecast, forecast_indices)

    #TODO: should I move this logic into the plot function?
    lower_bounds_residuals, upper_bounds_residuals = calculate_prediction_intervals_based_on_residuals(
    yhat_forecast_inverse, 
    std_devs_time_step, 
    confidence_level=0.95  # Ensure it's 95% confidence interval
    )
    last_date = combined_df.index[-1]
    date_range = pd.date_range(start=last_date, periods=FORECAST_RANGE, freq='W')

    # Use this to get the last actual price
    # Translate it
    # Translate the forecast feature name
    forecast_feature_tuple = translate_column_name(forecast_features[0])

    # Get the last actual price
    last_actual_price = combined_df[forecast_feature_tuple].iloc[-FORECAST_RANGE-1]

    # Call the plotting function
    plot_predictions_with_intervals_and_returns(
        yhat_forecast_inverse,  # This is y_pred in the function signature
        lower_bounds_residuals,  # This is lower_bound_res
        upper_bounds_residuals,  # This is upper_bound_res
        date_range,  # This is dates
        [forecast_feature_tuple[2]],  # This is feature_names, which should be ['Adj Close']
        last_actual_price
    )


