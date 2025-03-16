import pandas as pd
import numpy as np
from bokeh.plotting import figure
import yfinance as yf
from bokeh.models import HoverTool, ColumnDataSource, Select, Legend, NumeralTickFormatter
from bokeh.layouts import column
from bokeh.io import curdoc
from bokeh.models import Slider, Button
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')

# Function to calculate moving averages based on timeframe
def calculate_moving_averages(df, timeframe):
    sma_mapping = {"1M": (10, 20), "1Y": (50, 200), "5Y": (200, 800)}
    short_window, long_window = sma_mapping.get(timeframe, (10, 50))
    df['sma_short'] = df['price'].rolling(window=short_window).mean().astype(float)
    df['sma_long'] = df['price'].rolling(window=long_window).mean().astype(float)
    df.fillna(method="ffill", inplace=True)
    return df, short_window, long_window

# Function to predict prices with confidence intervals
def predict_prices(df, days_ahead=30, method='linear'):
    df_pred = df.copy()
    if method == 'linear':
        X = np.arange(len(df_pred)).reshape(-1, 1)
        y = df_pred['price'].values
        model = LinearRegression()
        model.fit(X, y)
        future_times = np.arange(len(df_pred), len(df_pred) + days_ahead).reshape(-1, 1)
        predictions = model.predict(future_times)
        
        # Improved confidence interval calculation using statsmodels OLS
        X_ols = sm.add_constant(X)
        ols_model = sm.OLS(y, X_ols).fit()
        pred_ols = ols_model.get_prediction(sm.add_constant(future_times))
        ci = pred_ols.conf_int()
        
        future_dates = pd.date_range(start=df_pred['timestamp'].iloc[-1], periods=days_ahead + 1, freq='D')[1:]
        pred_df = pd.DataFrame({
            'timestamp': future_dates,
            'price_pred': predictions,
            'lower_ci': ci[:, 0],
            'upper_ci': ci[:, 1]
        })
    elif method == 'arima':
        model = ARIMA(df_pred['price'], order=(5,1,0))
        model_fit = model.fit()
        forecast = model_fit.get_forecast(steps=days_ahead)
        predictions = forecast.predicted_mean
        ci = forecast.conf_int(alpha=0.05)
        future_dates = pd.date_range(start=df_pred['timestamp'].iloc[-1], periods=days_ahead + 1, freq='D')[1:]
        pred_df = pd.DataFrame({
            'timestamp': future_dates,
            'price_pred': predictions,
            'lower_ci': ci.iloc[:, 0],
            'upper_ci': ci.iloc[:, 1]
        })
    elif method == 'holt-winters':
        model = ExponentialSmoothing(df_pred['price'], trend="add", seasonal="add", seasonal_periods=7)
        model_fit = model.fit()
        predictions = model_fit.forecast(steps=days_ahead)
        resid = df_pred['price'] - model_fit.fittedvalues
        ci_width = 1.96 * np.std(resid)
        lower_ci = predictions - ci_width
        upper_ci = predictions + ci_width
        future_dates = pd.date_range(start=df_pred['timestamp'].iloc[-1], periods=days_ahead + 1, freq='D')[1:]
        pred_df = pd.DataFrame({
            'timestamp': future_dates,
            'price_pred': predictions,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci
        })
    elif method == 'prophet':
        # Rename columns and remove timezone from 'ds'
        df_prophet = df_pred.rename(columns={'timestamp': 'ds', 'price': 'y'})
        df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)  # Remove timezone
        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=days_ahead)
        forecast = model.predict(future)
        forecast = forecast.iloc[-days_ahead:]
        pred_df = pd.DataFrame({
            'timestamp': forecast['ds'].values,
            'price_pred': forecast['yhat'].values,
            'lower_ci': forecast['yhat_lower'].values,
            'upper_ci': forecast['yhat_upper'].values
        })
    return pred_df

# Function to fetch Bitcoin data from Yahoo Finance
def fetch_bitcoin_data(period):
    try:
        # Fetch BTC-USD data from Yahoo Finance
        btc = yf.Ticker("BTC-USD")
        df = btc.history(period=period, interval="1d")
        if not df.empty:
            # Reset index to get timestamp as a column
            df = df.reset_index()
            # Rename columns to match original structure
            df = df[['Date', 'Close']].rename(columns={'Date': 'timestamp', 'Close': 'price'})
            df['price'] = df['price'].astype(float)
            return df
    except Exception as e:
        print(f"Error fetching data from Yahoo Finance: {e}")
    return pd.DataFrame()

# Initial data fetch (default to 1Y)
df = fetch_bitcoin_data("1y")
df, short_window, long_window = calculate_moving_averages(df, "1Y")
pred_df = predict_prices(df, method='linear')

# Merge actual and predicted data, ensuring all columns are present
combined_df = pd.concat([df, pred_df], ignore_index=True)
combined_df = combined_df[['timestamp', 'price', 'sma_short', 'sma_long', 'price_pred', 'lower_ci', 'upper_ci']].fillna({'price_pred': np.nan, 'lower_ci': np.nan, 'upper_ci': np.nan})
source = ColumnDataSource(combined_df)

# Create Bokeh plot
p = figure(x_axis_type='datetime', title='Bitcoin Price Trends with Prediction', width=1000, height=500)

# Create line renderers
price_line = p.line('timestamp', 'price', source=source, color='blue', line_width=2, name='Actual Price')
sma_short_line = p.line('timestamp', 'sma_short', source=source, color='green', line_width=2, line_dash='dashed', name=f'SMA {short_window}')
sma_long_line = p.line('timestamp', 'sma_long', source=source, color='red', line_width=2, line_dash='dashed', name=f'SMA {long_window}')
pred_line = p.line('timestamp', 'price_pred', source=source, color='orange', line_width=2, line_dash='dotted', name='Prediction (Linear)')

# Add confidence interval band
pred_source = ColumnDataSource(pred_df)
ci_x = np.concatenate([pred_df['timestamp'], pred_df['timestamp'][::-1]])
ci_y = np.concatenate([pred_df['upper_ci'], pred_df['lower_ci'][::-1]])
ci_band = p.patch(ci_x, ci_y, color='orange', alpha=0.2, name='ci_band')

# Add legend
legend = Legend(items=[
    ("Actual Price", [price_line]),
    (f"SMA {short_window}", [sma_short_line]),
    (f"SMA {long_window}", [sma_long_line]),
    ("Prediction (Linear)", [pred_line]),
    ("95% Confidence Interval", [ci_band])
], location="top_left")
p.add_layout(legend)
p.legend.click_policy="hide"

# Format y-axis
p.yaxis.formatter = NumeralTickFormatter(format="$0,0")

# Add separate HoverTools for historical and predicted data
historical_hover = HoverTool(
    renderers=[price_line],  # Only apply the tooltip to the price line
    tooltips=[
        ('Date', '@timestamp{%F}'),
        ('Price', '$@price{0,0.00}'),
        ('SMA Short', '@sma_short{0,0.00}'),
        ('SMA Long', '@sma_long{0,0.00}')
    ],
    formatters={'@timestamp': 'datetime'},
    mode='vline'
)

prediction_hover = HoverTool(
    renderers=[pred_line, ci_band],
    tooltips=[
        ('Date', '@timestamp{%F}'),
        ('Predicted Price', '$@price_pred{0,0.00}'),
        ('Lower CI', '$@lower_ci{0,0.00}'),
        ('Upper CI', '$@upper_ci{0,0.00}'),
    ],
    formatters={'@timestamp': 'datetime'},
    mode='vline'
)

p.add_tools(historical_hover, prediction_hover)

# Dropdown menus
timeframe_select = Select(title="Select Timeframe:", value="1Y", options=["1M", "1Y", "5Y"])
method_select = Select(title="Prediction Method:", value="linear", options=["linear", "arima", "holt-winters", "prophet"])

# Create a slider for prediction horizon (in days)
prediction_horizon_slider = Slider(start=2, end=365, value=30, step=1, title="Prediction Horizon (Days)")

# Create the Apply Button
apply_button = Button(label="Apply", button_type="success")

# Update function to apply all changes (dropdowns and slider)
def on_apply_button_click():
    update_plot()

# Attach the Apply Button click event
apply_button.on_click(on_apply_button_click)

def update_plot():
    # Get the values from the dropdowns and slider
    timeframe = timeframe_select.value
    method = method_select.value
    prediction_horizon = prediction_horizon_slider.value

    # Map the selected timeframe to Yahoo Finance period
    period_map = {"1M": "30d", "1Y": "1y", "5Y": "5y"}
    
    # Fetch and process new data based on selected timeframe
    new_df = fetch_bitcoin_data(period_map[timeframe])
    new_df, short_window, long_window = calculate_moving_averages(new_df, timeframe)
    new_pred_df = predict_prices(new_df, method=method, days_ahead=prediction_horizon)
    
    # Merge actual and predicted data, ensuring all columns are present
    combined_df = pd.concat([new_df, new_pred_df], ignore_index=True)
    combined_df = combined_df[['timestamp', 'price', 'sma_short', 'sma_long', 'price_pred', 'lower_ci', 'upper_ci']].fillna({'price_pred': np.nan, 'lower_ci': np.nan, 'upper_ci': np.nan})
    source.data = ColumnDataSource.from_df(combined_df)
    
    # Update confidence interval band
    pred_source.data = ColumnDataSource.from_df(new_pred_df)
    ci_x = np.concatenate([new_pred_df['timestamp'], new_pred_df['timestamp'][::-1]])
    ci_y = np.concatenate([new_pred_df['upper_ci'], new_pred_df['lower_ci'][::-1]])

    # Remove old CI band and add new one
    p.renderers = [r for r in p.renderers if r.name != 'ci_band']
    ci_band = p.patch(ci_x, ci_y, color='orange', alpha=0.2, name='ci_band')
    
    # Update title with selected method and timeframe
    method_name = method.replace('-', ' ').title()
    p.title.text = f'Bitcoin Price Trends with {method_name} Prediction ({timeframe})'
    
    # Update legend items based on the selected method and moving averages
    p.legend.items = [
        ("Actual Price", [price_line]),
        (f"SMA {short_window}", [sma_short_line]),
        (f"SMA {long_window}", [sma_long_line]),
        (f"Prediction ({method_name})", [pred_line]),
        ("95% Confidence Interval", [ci_band])
    ]

# Layout includes the dropdowns, slider, and apply button
layout = column(timeframe_select, method_select, prediction_horizon_slider, apply_button, p)
curdoc().add_root(layout)