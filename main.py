from imports import *
from constants import *
from CNN_LSTM.evaluation import objective, backtest_and_evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data = {ticker: yf.download(ticker, start=start_date, end=end_date) for ticker in tickers}

# Fetch data for each indicator
# Fetch data for each indicator
indicator_data = {}
# Resample indicator data to weekly frequency and apply both lags
for indicator in indicators:
    indicator_data[indicator] = pdr.get_data_fred(indicator, start=start_date, end=end_date)
    indicator_data[indicator + '-2'] = indicator_data[indicator].shift(2)  # Lagging the indicator by 2 periods
    indicator_data[indicator + '-91'] = indicator_data[indicator].shift(91)  # Lagging the indicator by 91 periods
    indicator_data[indicator + '-2'] = indicator_data[indicator + '-2'].resample('W').last()
    indicator_data[indicator + '-91'] = indicator_data[indicator + '-91'].resample('W').last()

# Create a new DataFrame to hold the required features
stock_data = []
for key in data.keys():
    df = data[key]
    df['vwp'] = df['Adj Close'] * df['Volume']
    df['p-1'] = df['Adj Close'].shift(1)  # Previous day's price
    df = df[['Adj Close', 'vwp', 'p-1', 'Volume']]
    df = df.resample('W').last()  # Resample to weekly data
    df['date_feature'] = (df.index - df.index[0]).days
    df['ticker'] = key
    stock_data.append(df)

# Combine stock data into a single DataFrame
combined_stock_data = pd.concat(stock_data)

# Pivot the data to have a multi-level column index
pivot_stock_df = combined_stock_data.pivot_table(index='Date', columns='ticker', values=['Adj Close', 'vwp', 'p-1', 'Volume', 'date_feature'])
pivot_stock_df.columns = pd.MultiIndex.from_tuples([('stock', col[1], col[0]) for col in pivot_stock_df.columns])

# Create a DataFrame for the indicators with a multi-level column index
pivot_indicator_df = pd.DataFrame(index=pivot_stock_df.index)
for indicator in indicators:
    pivot_indicator_df[('fred', indicator, 'value-2')] = indicator_data[indicator + '-2']  # Adjust column name to indicate the 2-day shift
    pivot_indicator_df[('fred', indicator, 'value-91')] = indicator_data[indicator + '-91']  # Adjust column name to indicate the 91-day shift

# Combine stock and indicator data
combined_df = pd.concat([pivot_stock_df, pivot_indicator_df], axis=1)
combined_df.dropna(inplace=True)

# Rest of the code remains the same
trial_results = []

# Run the study

# Create the study
study = optuna.create_study(direction="minimize")

# Optimize with the additional parameters
study.optimize(functools.partial(objective, 
                                 combined_df=combined_df, 
                                 forecast_features=forecast_features,
                                 trial_results=trial_results,
                                 LOOK_BACK=LOOK_BACK,
                                 FORECAST_RANGE=FORECAST_RANGE,
                                NUM_EPOCHS=NUM_EPOCHS), 
               n_trials=N_TRIALS)

# Convert results to a DataFrame

results_df = pd.DataFrame(trial_results)

significant_features = []

for feature in features:
    mape_with_feature = results_df[results_df[feature] == True]['mape']
    mape_without_feature = results_df[results_df[feature] == False]['mape']
    
    t_stat, p_value = stats.ttest_ind(mape_with_feature, mape_without_feature)
    
    if p_value < 0.05:  # Using 0.05 as the significance level
        significant_features.append(feature)
        print(f"{feature} is significant with p-value: {p_value}")

# Sort trials by MAPE
results_df = results_df.sort_values('trial_number')

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(results_df['trial_number'], results_df['mape'], marker='o')

# Highlight points where significant features were included
for feature in significant_features:
    feature_trials = results_df[results_df[feature] == True]
    plt.scatter(feature_trials['trial_number'], feature_trials['mape'], 
                label=feature, s=100, alpha=0.6)

plt.title('MAPE Reduction Over Optuna Trials')
plt.xlabel('Trial Number')
plt.ylabel('MAPE')
plt.legend()
plt.grid(True, alpha=0.3)

# Add text annotation for best MAPE
best_trial = results_df.loc[results_df['mape'].idxmin()]
plt.annotate(f"Best MAPE: {best_trial['mape']:.2f}", 
             xy=(best_trial['trial_number'], best_trial['mape']), 
             xytext=(5, 5), textcoords='offset points')
plt.savefig('trial_mape.png')
#plt.show()

# Print the best trial information
print("\nBest trial:")
print(f"  MAPE: {best_trial['mape']:.2f}")
print("  Features:")
for feature in features:
    if best_trial[feature]:
        print(f"    - {feature}")

# The rest of your code remains unchanged
best_trial = study.best_trial
print("Best trial:")
print(f"  Value: {best_trial.value}")
print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

features = [
    ('stock', 'GOOGL', 'Adj Close'), 
    ('stock', 'GOOGL', 'Volume'), 
    ('stock', 'GOOGL', 'vwp'), 
    ('stock', 'GOOGL', 'p-1'),
    ('fred', 'T10Y3M', 'value-2'),
    ('fred', 'T10Y3M', 'value-91'),
    ('fred', 'EFFR', 'value-2'),
    ('fred', 'EFFR', 'value-91')
]
forecast_features = [('stock', 'GOOGL', 'Adj Close')]
backtest_and_evaluate(study, combined_df, MinMaxScaler(), features, forecast_features, NUM_EPOCHS=NUM_EPOCHS)