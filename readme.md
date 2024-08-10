Inspired by
#https://towardsdatascience.com/cnn-lstm-based-models-for-multiple-parallel-input-and-multi-step-forecast-6fe2172f7668

Tree-structured Parzen Estimator

optimal lags identified via max absolute significant Median Correlation over moving windows


Features
* Significant Median correlations over moving windows used to identify lags to test for
* t-tests used to determine whether or not mape reduced significantly enough to include a feature.
* inflection point where csum roc descends