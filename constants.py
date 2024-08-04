from imports import *

# constants.py
NUM_SIMULATIONS = 3
N_TRIALS = 5
NUM_EPOCHS = 20
LOOK_BACK = 156
FORECAST_RANGE = 13

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
end_date = datetime.now()
start_date = end_date - timedelta(days=365*5)  # 5 years of data

NUM_SIMULATIONS = 3  # or any other number you prefer
N_TRIALS = 5
NUM_EPOCHS = 20
LOOK_BACK = 156
FORECAST_RANGE = 13

# Download data for tickers
tickers = ['GOOGL', 'SPY']

# Define your parameters
forecast_features = ['stock.GOOGL.Adj Close']

indicators = ['T10Y3M', 'EFFR']

# Perform t-tests for each feature
features = ['include_volume', 'include_vwp', 'include_spy', 'include_T10Y3M', 'include_EFFR', 'include_p1']

trial_results = []