# Energy Market Analysis & Forecasting Using ARIMA

## Overview
This project retrieves and analyzes **U.S. energy market data** from the **Energy Information Administration (EIA) API** and applies **time series forecasting (ARIMA)** to predict future energy consumption trends. The project leverages **Python, data visualization, and machine learning** techniques to model and forecast energy consumption patterns. 
## Key Features
- **Real-Time Data Retrieval** - Obtains energy consumption data via the **EIA API**.
- **Data Cleaning & Preprocessing** - Handles missing values, format dates, and filters relevant data.
- **Exploratory Data Analysis (EDA)** - Uses **Matplotlib, Seaborn, and Plotly** for trend visualization.
- **ARIMA-Based Forecasting** - Predicts future energy consumption trends using **time series** modeling.
- **Model Performance Evaluation** - Assesses forecast accuracy using **Root Mean Squared Error (RSME)**.

## Outline
- [Installation](#installation)
- [Requirements](#requirements)
- [API Integration & Data Collection](#API_Integration_&_Data_Collection)
- [Data Visualization](#Data_Visualization)
- [Functionality](#functionality)
- [Future Enhancements](#Future_Enhancements)
- [License](#license)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/daniela1484/Energy-Price-Prediction-and-Analysis.git
   cd Energy-Price-Prediction-and-Analysis
   ```

2. Create a virtual environment (this is optional but recommended):
   ```
   python -m venv env
   source env/bin/activate   # Mac/Linux
   env\Scripts\activate      # Windows
   ```

3. Install the required packages:
   ```
   pip install pandas numpy requests matplotlib seaborn plotly statsmodels scikit-learn
   ```
## Requirements
- Python 3.13.2
- Required Libraries:
  - `pandas`
  - `numpy`
  - `requests`
  - `matplotlib`
  - `seaborn`
  - `plotly`
  - `statsmodels`
  - `scikit-learn`
  
## API Integration & Data Collection
**Obtaining Energy Market Data from EIA API**
The script retrieves **annual energy consumption data** (in trillion Btu) from the **EIA API** for the years 2023-2025.
   ```
   import requests
import pandas as pd

EIA_API_URL = "https://api.eia.gov/v2/aeo/2023/data/?frequency=annual&data[0]=value&start=2023&end=2025&api_key=YOUR_API_KEY"

def energy_data():
    response = requests.get(EIA_API_URL)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data['response']['data'])
        df['value'] = pd.to_numeric(df["value"], errors='coerce').fillna(0)
        df['date'] = pd.to_datetime(df["period"], format="%Y")
        return df
    else:
        print(f"Error fetching data: {response.status_code}")
        return None

   ```
## Data Visualization
**Energy Consumption Over Time**
   ```
  import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))
sns.lineplot(x=df['date'], y=df['value'], marker='o', label="Energy Consumption")
plt.ylabel("Value / Consumption")
plt.xlabel("Year")
plt.title("Energy Consumption Trends")
plt.legend()
plt.show()
   ```
> Generates an interactive time-series plot showing historical energy consumption.

## ARIMA Time Series Forecasting
**Step 1: Train-Test Split**
```
train_size = int(len(df) * 0.8)
train, test = df['value'][0:train_size], df['value'][train_size:]
```
> Uses **80% of data for training, 20% for testing**.

**Step 2: Build & Train ARIMA Model**
```
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train, order=(5, 1, 0))  # ARIMA(p,d,q) parameters
model_fit = model.fit()
forecast = model_fit.forecast(steps=len(test))
```
> Uses **ARIMA (5, 1, 0)** for time series modeling.

**Step 3: Model Performance Evaluation**
```
from sklearn.metrics import mean_squared_error
import numpy as np

rmse = np.sqrt(mean_squared_error(test, forecast))
print(f"Root Mean Squared Error: {rmse}")
```
> Evaluates model accuracy using **RSME**.

**Step 4: Visualizing Predictions vs. Actual Data**
```
plt.figure(figsize=(10, 5))
sns.lineplot(x=df['date'].iloc[train_size:], y=test, label="Actual Values")
sns.lineplot(x=df['date'].iloc[train_size:], y=forecast, label="Predicted Values")
plt.xlabel("Year")
plt.ylabel("Energy Consumption")
plt.title("Energy Consumption: Predicted vs Actual")
plt.legend()
plt.show()
```
> Compares actual vs. predicted values to assess model performance.

## Functionality
1. **Fetch Data**: The script connects to the EIA API and retrieves energy consumption data.
2. **Process Data**: Converts the fetched data into a Pandas DataFrame, filters for units in trillion BTU, and converts relevant columns to appropriate data types.
3. **Visualize Data**: Generates plots to compare historical energy values and consumption.
4. **Forecasting**: Implements an ARIMA model to predict future energy consumption based on historical data.
5. **Evaluate Model**: Computes RMSE to measure the accuracy of the predictions and displays the results.

## Future Enhancements
- Extend forecasting horizon beyond 2025
- Use advanced machine learning models (LSTMs, XGBoost) for better predictions
- Develop a web dashboard for real-time energy data monitoring

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
