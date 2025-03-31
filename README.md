# Energy Price Prediction and Analysis

This Python script retrieves energy market data from the U.S. Energy Information Administration (EIA) API, processes it, and performs time series forecasting using an ARIMA model. It also visualizes historical energy consumption and predictions using Matplotlib and Seaborn.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Functionality](#functionality)
- [License](#license)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/daniela1484/Energy-Price-Prediction-and-Analysis.git
   cd Energy-Price-Prediction-and-Analysis
   ```

3. Create a virtual environment (this is optional but recommended):
   ```
   python -m venv env
   ```
3. Activate the virtual environment:
   - On Windows:
   ```
     env\Scripts\activate
   ```
   - On macOS/Linux:
   ```
     source env/bin/activate
   ```

4. Install the required packages:
   ```
   pip install pandas numpy requests matplotlib seaborn plotly statsmodels scikit-learn
   ```
## Usage

1. Replace the placeholder API key in the script: 
   ```
   EIA_API_URL = "https://api.eia.gov/v2/aeo/2023/data/?frequency=annual&data[0]=value&start=2023&end=2025&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000&api_key=GcdSVebgcbQ442EUDFnvdJNj39wQ3UdpDk6PkhUs"
   ```
3. Run the script:
   ```
   python energy_consumption_analysis.py
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
 
## Functionality
1. **Fetch Data**: The script connects to the EIA API and retrieves energy consumption data.
2. **Process Data**: Converts the fetched data into a Pandas DataFrame, filters for units in trillion BTU, and converts relevant columns to appropriate data types.
3. **Visualize Data**: Generates plots to compare historical energy values and consumption.
4. **Forecasting**: Implements an ARIMA model to predict future energy consumption based on historical data.
5. **Evaluate Model**: Computes RMSE to measure the accuracy of the predictions and displays the results.

## Contributing
Any contributions are welcomed! Please feel free to submit a pull request or open an issue for any improvements or bug fixes. 

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
