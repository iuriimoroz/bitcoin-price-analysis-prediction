# Bitcoin Price Trend Analysis and Prediction

## Project Overview
This project provides an interactive dashboard for analyzing Bitcoin price trends and generating price predictions using various forecasting models. The application visualizes historical Bitcoin price data alongside different moving averages and displays predictions with confidence intervals.

## Features
- **Real-time Data**: Fetches Bitcoin price data from Yahoo Finance
- **Interactive Timeframes**: Analyze price trends across multiple timeframes (1 Month, 1 Year, 5 Years)
- **Technical Indicators**: Displays Simple Moving Averages with dynamic periods based on the selected timeframe
- **Multiple Prediction Models**: 
  - Linear Regression
  - ARIMA (AutoRegressive Integrated Moving Average)
  - Holt-Winters Exponential Smoothing
  - Facebook Prophet
- **Confidence Intervals**: All predictions include 95% confidence intervals
- **Adjustable Prediction Horizon**: Customize forecast length from 2 to 365 days
- **Interactive Controls**: User-friendly interface with dropdown selectors and sliders

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Required Packages
```bash
pip install pandas numpy bokeh yfinance scikit-learn statsmodels prophet
```

### Running the Dashboard
1. Clone this repository:
```bash
git clone https://github.com/yourusername/bitcoin-price-analysis-prediction.git
cd bitcoin-price-analysis-prediction
```

2. Launch the Bokeh server:
```bash
bokeh serve --show Capstone2.py
```

3. The dashboard will open in your default web browser

## How to Use
1. **Select Timeframe**: Choose between 1 Month, 1 Year, or 5 Years of historical data
2. **Choose Prediction Model**: Select from Linear, ARIMA, Holt-Winters, or Prophet algorithms
3. **Adjust Prediction Horizon**: Use the slider to set how many days ahead to predict
4. **Apply Changes**: Click the "Apply" button to update the visualization
5. **Interactive Exploration**: Hover over the chart to see detailed data points
6. **Toggle Visibility**: Click on legend items to show/hide specific series

## Project Structure
- `Capstone2.py` - The main Python script containing the dashboard application
- `Capstone 2_ Bitcoin Price Trend Analysis and Prediction.ipynb` - Jupyter notebook with exploratory analysis and model development

## Technical Details
- **Data Source**: Yahoo Finance API via the yfinance package
- **Visualization**: Built with Bokeh interactive visualization library
- **Moving Averages**: Dynamically calculated based on the selected timeframe
  - 1M: 10-day and 20-day SMAs
  - 1Y: 50-day and 200-day SMAs
  - 5Y: 200-day and 800-day SMAs
- **Prediction Methods**:
  - Linear Regression with confidence intervals from statsmodels
  - ARIMA(5,1,0) model for time series forecasting
  - Holt-Winters triple exponential smoothing with additive trend and seasonality
  - Facebook Prophet with default parameters

## Future Improvements
- Add more technical indicators (RSI, MACD, Bollinger Bands)
- Implement backtesting capabilities to evaluate prediction accuracy
- Add trading signal generation based on technical indicators
- Incorporate sentiment analysis from social media or news sources
- Support for additional cryptocurrencies

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Yahoo Finance for providing the data API
- The developers of Pandas, Bokeh, and the various forecasting libraries
