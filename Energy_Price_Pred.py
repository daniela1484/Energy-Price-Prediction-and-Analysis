import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

EIA_API_URL = "https://api.eia.gov/v2/aeo/2023/data/?frequency=annual&data[0]=value&start=2023&end=2025&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000&api_key=GcdSVebgcbQ442EUDFnvdJNj39wQ3UdpDk6PkhUs"

# Obtain the Energy Market Data from EIA API
def energy_data():
    response = requests.get(EIA_API_URL)
    if response.status_code == 200:
        data = response.json()
        if 'response' in data and 'data' in data['response']:
            data_values = data['response']['data'] # Access 'data' inside 'response'
            # Convert to DataFrame
            df = pd.DataFrame(data_values)
            # Convert 'value' column to numeric
            df['value'] = pd.to_numeric(df["value"], errors='coerce').fillna(0)
            # Convert 'period' to datetime
            df['date'] = pd.to_datetime(df["period"], format="%Y")
            # Filter for "trillion Btu" only
            df_filter = df[df['unit'] == "trillion Btu"]
            # Return the DataFrame and its relevant columns
            return df, df['value'], df_filter #df['date']
        else:
            print("No 'data' key found in 'response'.")

    else:
        print(f"Error fetching data from EIA API: {response.status_code}")
        return None
    
# Call the fcn to get the data
df, value, filter_energy = energy_data()

# Check if data was successfully retrieved b4 using it
if df is not None:
    print(df.head()) 
else:
    print("Failed to retrive the data.")

# Plot historical energy prices
plt.figure(figsize=(10, 5))
sns.lineplot(x=df['date'], y=df['value'], marker='o', label= "Energy Value")
plt.ylabel("Value / Consumption")
plt.xlabel("Year")
plt.title("Comparison of Energy value and Energy Consumption")
plt.legend()
plt.show()

# ARIMA Model for Time Series Forecasting
train_size = int(len(df) * 0.8) # 80% for training
train, test = value[0:train_size], value[train_size:]

model = ARIMA(train, order=(5, 1, 0)) # ARIMA(p,d,q) parameters
model_fit = model.fit()
forecast = model_fit.forecast(steps=len(test))

# Evaluate the model performance
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f"Root Mean Squared Error: {rmse}")

# Plot pred vs actual
test_dates = df['date'].iloc[train_size:]
plt.figure(figsize=(10, 5))
sns.lineplot(x=test_dates, y=test, label="Actual Values")
sns.lineplot(x=test_dates, y=forecast, label="Predicted Values")
plt.xlabel("Year")
plt.ylabel("Energy Consumption")
plt.title("Energy Consumption: Predicted vs Actual")
plt.legend()
plt.show()
