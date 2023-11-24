# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load your dataset
# Replace 'your_dataset.csv' with the actual file path or URL to your dataset
data = pd.read_csv('your_dataset.csv')

# Assuming 'Date' is a string, convert it to a datetime object
data['Date'] = pd.to_datetime(data['Date'])

# Sort the data by date
data.sort_values('Date', inplace=True)

# Create a time series plot to visualize the data
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Sales'])
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data[0:train_size], data[train_size:]

# Train a simple machine learning model (Random Forest Regressor)
X_train, y_train = np.arange(len(train)).reshape(-1, 1), train['Sales']
X_test, y_test = np.arange(len(train), len(data)).reshape(-1, 1), test['Sales']

# Initialize and train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
rf_predictions = rf_model.predict(X_test)

# Evaluate the performance of the model
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
print(f"Random Forest RMSE: {rf_rmse}")

# Train a time series forecasting model (Exponential Smoothing)
ts_model = ExponentialSmoothing(train['Sales'], seasonal='add', seasonal_periods=12)
ts_model_fit = ts_model.fit()
ts_predictions = ts_model_fit.forecast(len(test))

# Evaluate the performance of the time series model
ts_rmse = np.sqrt(mean_squared_error(test['Sales'], ts_predictions))
print(f"Time Series (Exponential Smoothing) RMSE: {ts_rmse}")

# Plot the actual vs. predicted values for both models
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Sales'], label='Actual Sales')
plt.plot(test['Date'], rf_predictions, label='Random Forest Predictions')
plt.plot(test['Date'], ts_predictions, label='Time Series Predictions')
plt.title('Sales Prediction Comparison')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()
