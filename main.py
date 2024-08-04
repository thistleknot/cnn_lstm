from imports import *
from constants import *
from CNN_LSTM.evaluation import objective, backtest_and_evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data = {ticker: yf.download(ticker, start=start_date, end=end_date) for ticker in tickers}

# Fetch data for each indicator
indicator_data = {}
for indicator in indicators:
    ind_data = pdr.get_data_fred(indicator, start=start_date, end=end_date)
    # Interpolate missing data
    #TODO: Data leakage, use EMA of ffill
    ind_data = ind_data.interpolate()
    
    # Assign the interpolated data to the dictionary
    indicator_data[indicator] = ind_data
    #TODO: use last business day as defined from nyse_dates
    indicator_data[indicator + '-2'] = ind_data.shift(2).resample('W').last()
    indicator_data[indicator + '-91'] = ind_data.shift(91).resample('W').last()

# Create a new DataFrame to hold the required features
stock_data = []
for key in data.keys():
    df = data[key]
    df['vwp'] = df['Adj Close'] * df['Volume']
    df['p-1'] = df['Adj Close'].shift(1)
    df = df[['Adj Close', 'vwp', 'p-1', 'Volume']].resample('W').last()
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
    pivot_indicator_df[('fred', indicator, 'value-2')] = indicator_data[indicator + '-2']
    pivot_indicator_df[('fred', indicator, 'value-91')] = indicator_data[indicator + '-91']

# Combine stock and indicator data
combined_df = pd.concat([pivot_stock_df, pivot_indicator_df], axis=1)
combined_df.dropna(inplace=True)
# Create binary flags for Q1, Q2, Q3, and Q4 directly using index month
combined_df[('date_', 'quarter', 'Q1')] = combined_df.index.to_series().apply(lambda x: 1 if x.month in [1, 2, 3] else 0)
combined_df[('date_', 'quarter', 'Q2')] = combined_df.index.to_series().apply(lambda x: 1 if x.month in [4, 5, 6] else 0)
combined_df[('date_', 'quarter', 'Q3')] = combined_df.index.to_series().apply(lambda x: 1 if x.month in [7, 8, 9] else 0)
combined_df[('date_', 'quarter', 'Q4')] = combined_df.index.to_series().apply(lambda x: 1 if x.month in [10, 11, 12] else 0)

# Run the study
study = optuna.create_study(direction="minimize")
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
for feature in include_flags:
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
for feature in include_flags:
    if best_trial[feature]:
        print(f"    - {feature}")

# The rest of your code remains unchanged
best_trial = study.best_trial
print("Best trial:")
print(f"  Value: {best_trial.value}")
print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

backtest_and_evaluate(study, combined_df, MinMaxScaler(), features, forecast_features, NUM_EPOCHS=NUM_EPOCHS)