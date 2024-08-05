from imports import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
end_date = datetime.now()
start_date = end_date - timedelta(days=365*5)  # 5 years of data

debug = False

NUM_SIMULATIONS = 3  # or any other number you prefer
N_TRIALS = 5
NUM_EPOCHS = 20
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
    'stock.SPY.vwp', 
    'stock.SPY.Volume', 
    'stock.SPY.Adj Close', 
    'fred.T10Y3M.value',
    'fred.EFFR.value',
    'date.quarter.Q1',
    'date.quarter.Q2',
    'date.quarter.Q3',
    'date.quarter.Q4'
]

# Define indicators
indicators = ['T10Y3M', 'EFFR']

trial_results = []