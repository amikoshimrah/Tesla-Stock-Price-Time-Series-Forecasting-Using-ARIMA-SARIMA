import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Tesla Forecasting", layout="wide")

# GitHub raw CSV URL (replace with your own if needed)
CSV_URL = "https://raw.githubusercontent.com/amikoshimrah/Tesla-Stock-Price-Time-Series-Forecasting-Using-ARIMA-SARIMA/main/TSLA.csv"


# Load historical data from GitHub
@st.cache_data
def load_historical_data():
    df = pd.read_csv(CSV_URL)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df[['Close']].dropna()

# Load models with their corresponding last date
@st.cache_resource
def load_models():
    with open("arima_model_tsa.pkl", "rb") as f_arima:
        arima_model, arima_last_date = pickle.load(f_arima)
    with open("sarima_model_tsa.pkl", "rb") as f_sarima:
        sarima_model, sarima_last_date = pickle.load(f_sarima)
    return {
        "ARIMA": (arima_model, arima_last_date),
        "SARIMA": (sarima_model, sarima_last_date)
    }

# Load data and models
historical_df = load_historical_data()
models = load_models()

# Sidebar inputs
with st.sidebar:
    st.header("📊 Forecast Settings")
    model_choice = st.selectbox("Select Model", list(models.keys()))
    forecast_months = st.slider("Forecast Period (months)", 1, 36, 12)
    st.write("🔵 Actual: Historical TSLA Close Price")
    st.write("🔴 Forecast: Model Prediction")

# Page Title
st.title("📈 Tesla Stock Price Forecasting")
st.markdown(f"Forecasting with **{model_choice}** model")

# Forecast logic
if st.button("🔮 Generate Forecast"):
    model, last_date = models[model_choice]

    # Generate forecast
    forecast = model.forecast(steps=forecast_months)

    # Future date range
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=forecast_months,
        freq='M'
    )

    # Create forecast DataFrame
    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast}).set_index('Date')

    # Plot actual + forecast
    st.subheader("📉 Tesla Stock Price Forecast")
    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot historical data (blue)
    ax.plot(historical_df.index, historical_df['Close'], label='Actual (Historical)', color='blue')

    # Plot forecast data (red)
    ax.plot(forecast_df.index, forecast_df['Forecast'], label=f'{model_choice} Forecast', color='red')

    ax.set_title(f"{model_choice} Forecast for Next {forecast_months} Months")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # Show forecast table
    st.subheader("🔍 Forecast Table")
    st.dataframe(forecast_df.round(2))

# Optionally show historical table
with st.expander("📜 View Historical Data"):
    st.dataframe(historical_df.tail(50).style.format({"Close": "${:,.2f}"}))
