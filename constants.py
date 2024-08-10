from imports import *
from functions import create_feature_list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
end_date = datetime.now()
start_date = end_date - timedelta(days=365*5)  # 5 years of data

debug = False

NUM_SIMULATIONS = 10  # or any other number you prefer
#should be beased on len features
N_TRIALS = 40
NUM_EPOCHS = 50
LOOK_BACK = 156
FORECAST_RANGE = 13

sector_etfs = [
    "XLK",
    "XLV",
    "XLF",
    "XLY",
    "XLP",
    "XLE",
    "XLI",
    "XLB",
    "XLU",
    "XLRE",
    "XLC"
]

other_etfs = [
    'VIX'
]

# Download data for tickers
#vwp, volume, adj_close
tickers_all = ['GOOGL', 'SPY']
tickers_price = [*sector_etfs,*other_etfs]

tickers_combined = [*tickers_all,*tickers_price]

# Define your parameters
forecast_features = ['stock.GOOGL.Adj Close']

# Define indicators
indicators = ['T10Y3M', 'EFFR']

# Base features to always include
base_features = [
    *forecast_features,
    'date.quarter.Q1',
    'date.quarter.Q2',
    'date.quarter.Q3',
    'date.quarter.Q4'
]

# Generate the final features list
features = create_feature_list(tickers_all, tickers_price, indicators, base_features)

# Output the final features list for inspection
print(features)
#ffill, balance sheet info, fundamentals
#inflation, gdp, unrate, confidence
#how would I turn this into a graph?

trial_results = []