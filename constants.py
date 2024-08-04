from imports import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
end_date = datetime.now()
start_date = end_date - timedelta(days=365*5)  # 5 years of data

debug = False

NUM_SIMULATIONS = 10  # or any other number you prefer
N_TRIALS = 20
NUM_EPOCHS = 50
LOOK_BACK = 156
FORECAST_RANGE = 13

# Download data for tickers
tickers = ['GOOGL', 'SPY']

# Define your parameters
forecast_features = ['stock.GOOGL.Adj Close']

features = [
    'stock.GOOGL.Adj Close', 
    'stock.GOOGL.Volume', 
    'stock.GOOGL.vwp', 
    'stock.GOOGL.p-1',
    'fred.T10Y3M.value-2',
    'fred.T10Y3M.value-91',
    'fred.EFFR.value-2',
    'fred.EFFR.value-91',
    'date.quarter.Q1',
    'date.quarter.Q2',
    'date.quarter.Q3',
    'date.quarter.Q4'
]

# Define indicators
indicators = ['T10Y3M', 'EFFR']

# Generate include flags from the features list
include_flags = [f'include.{feature}' for feature in features[1:]]  # Skip the first feature as it is always included

trial_results = []