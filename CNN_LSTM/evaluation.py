import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from .training import train_model
from .CNN_TS import LSTMEncoderDecoderAttention
from imports import *
from constants import *
from .visualization import plot_predictions_with_intervals

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

def calculate_prediction_intervals_based_on_residuals(predictions, std_devs):
    lower_bounds = predictions - std_devs
    upper_bounds = predictions + std_devs
    return lower_bounds, upper_bounds

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

def objective(trial, combined_df, forecast_features, trial_results, LOOK_BACK, FORECAST_RANGE, NUM_EPOCHS):
    # Select features to include
    include_volume = trial.suggest_categorical("include_volume", [True, False])
    include_vwp = trial.suggest_categorical("include_vwp", [True, False])
    include_spy = trial.suggest_categorical("include_spy", [True, False])
    include_T10Y3M = trial.suggest_categorical("include_T10Y3M", [True, False])
    include_EFFR = trial.suggest_categorical("include_EFFR", [True, False])
    include_p1 = trial.suggest_categorical("include_p1", [True, False])  # New line
    
    # Construct the feature list based on the trial's suggestions
    features = ['Adj Close']
    if include_volume:
        features.append('Volume')
    if include_vwp:
        features.append('vwp')
    if include_spy:
        features.append('SPY')
    if include_T10Y3M:
        features.append('T10Y3M')
    if include_EFFR:
        features.append('EFFR')
    if include_p1:
        features.append('p-1')  # New line
    
    # Normalize and prepare data
    selected_columns = [col for col in combined_df.columns if any(feature in col for feature in features)]
    data_subset = combined_df[selected_columns]
    
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
        'include_volume': include_volume,
        'include_vwp': include_vwp,
        'include_spy': include_spy,
        'include_T10Y3M': include_T10Y3M,
        'include_EFFR': include_EFFR,
        'include_p1': include_p1  # New line
    })
    
    return average_mape

def backtest_and_evaluate(study, combined_df, scaler, features, forecast_features, LOOK_BACK=156, FORECAST_RANGE=13, NUM_EPOCHS=20):
    selected_columns = [col for col in combined_df.columns if col in features]
    data_subset = combined_df[selected_columns]
    
    scaled_data = scaler.fit_transform(data_subset)
    scaled_data = pd.DataFrame(scaled_data, columns=data_subset.columns, index=data_subset.index)
    
    # Convert multi-level column names to string
    scaled_data.columns = ['.'.join(col) if isinstance(col, tuple) else col for col in scaled_data.columns]
    feature_mapping = {col: idx for idx, col in enumerate(scaled_data.columns)}
    
    # Convert forecast_features to string format if they're tuples
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
    
    residuals_time_step = y_train_inverse_time_step - yhat_inverse_time_step
    std_devs_time_step = np.std(residuals_time_step, axis=0).squeeze()
    
    last_values = scaled_data.values[-LOOK_BACK:]
    X_last = np.expand_dims(last_values, axis=0)
    X_last = torch.tensor(X_last, dtype=torch.float32).to(device)
    
    model.eval()
    with torch.no_grad():
        yhat_forecast = model(X_last).cpu().numpy()
    
    yhat_forecast = yhat_forecast.reshape(-1, len(forecast_indices))
    yhat_forecast_inverse, _ = inverse_transform(scaler, np.zeros_like(yhat_forecast), yhat_forecast, forecast_indices)
    
    yhat_forecast_inverse_last = yhat_forecast_inverse
    
    std_dev_constant = 0.67
    lower_bounds_original = yhat_forecast_inverse_last - std_dev_constant * yhat_forecast_inverse_last
    upper_bounds_original = yhat_forecast_inverse_last + std_dev_constant * yhat_forecast_inverse_last
    
    lower_bounds_residuals, upper_bounds_residuals = calculate_prediction_intervals_based_on_residuals(yhat_forecast_inverse_last, std_devs_time_step)
    
    last_date = combined_df.index[-1]
    date_range = pd.date_range(start=last_date, periods=FORECAST_RANGE, freq='W')
    
    plot_predictions_with_intervals(yhat_forecast_inverse_last, lower_bounds_residuals, upper_bounds_residuals, date_range, forecast_features)

