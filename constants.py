from imports import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
end_date = datetime.now()
start_date = end_date - timedelta(days=365*5)  # 5 years of data

debug = False

NUM_SIMULATIONS = 3  # or any other number you prefer
N_TRIALS = 10
NUM_EPOCHS = 20
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

sector_etf_features = [
    "stock.XLK.AdjClose",
    "stock.XLV.AdjClose",
    "stock.XLF.AdjClose",
    "stock.XLY.AdjClose",
    "stock.XLP.AdjClose",
    "stock.XLE.AdjClose",
    "stock.XLI.AdjClose",
    "stock.XLB.AdjClose",
    "stock.XLU.AdjClose",
    "stock.XLRE.AdjClose",
    "stock.XLC.AdjClose"
]

other_features = [
    "stock.VIX.AdjClose"
]



# Define indicators
indicators = ['T10Y3M', 'EFFR']

# Base features to always include
base_features = [
    'date.quarter.Q1',
    'date.quarter.Q2',
    'date.quarter.Q3',
    'date.quarter.Q4',
    *sector_etf_features,
    *other_features
]
def generate_features_for_tickers(tickers):
    """Generate stock-related features based on provided tickers."""
    features = []
    for ticker in tickers:
        features.append(f'stock.{ticker}.Adj Close')
        features.append(f'stock.{ticker}.Volume')
        features.append(f'stock.{ticker}.vwp')
    return features

def expand_indicators(indicators):
    """Expand indicators into their .value representation."""
    return [f'fred.{indicator}.value' for indicator in indicators]

def create_feature_list(tickers_all, tickers_price, indicators, base_features):
    """Create a list of all features including ticker-related, indicators, and base features."""
    # Generate ticker-specific features for tickers_all
    ticker_features = generate_features_for_tickers(tickers_all)
    
    # Generate ticker-specific features for tickets_price (sector ETFs and other ETFs)
    price_features = generate_features_for_tickers(tickers_price)
    
    # Expand indicators
    expanded_indicators = expand_indicators(indicators)
    
    # Combine all features
    features = ticker_features + price_features + expanded_indicators + base_features
    
    return features

# Generate the final features list
features = create_feature_list(tickers_all, tickers_price, indicators, base_features)

# Output the final features list for inspection
print(features)
print(features)
#ffill, balance sheet info, fundamentals
#inflation, gdp, unrate, confidence
#how would I turn this into a graph?

trial_results = []