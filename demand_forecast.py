import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load your data
# Replace 'your_demand_data.csv' with the path to your data file
data = pd.read_csv('shipment_data.csv', parse_dates=['Date'], index_col='Date')

# Fit an ARIMA model
model = ARIMA(data['Demand'], order=(1, 1, 1))  # Adjust the order as needed
model_fit = model.fit()

# Make predictions
forecast = model_fit.forecast(steps=10)  # Forecast the next 10 time periods

# Plot the results
plt.plot(data['Demand'], label='Historical Demand')
plt.plot(forecast, label='Forecasted Demand', color='red')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Demand')
plt.title('Demand Forecasting')
plt.show()
