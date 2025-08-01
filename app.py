import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

# Load models
@st.cache_resource
def load_models():
    with open("arima_model_tsa.pkl", "rb") as f_arima:
        arima_model = pickle.load(f_arima)
    with open("sarima_model_tsa.pkl", "rb") as f_sarima:
        sarima_model = pickle.load(f_sarima)
    return arima_model, sarima_model

arima_model, sarima_model = load_models()

# App Title
st.title("📈 Tesla Stock Price Forecasting")
st.markdown("Using ARIMA and SARIMA Models")

# Forecast period input
n_months = st.slider("Select forecast period (months)", 12, 60, 24)

# Forecast Button
if st.button("Generate Forecast"):

    # Forecast
    forecast_arima = arima_model.forecast(steps=n_months)
    forecast_sarima = sarima_model.forecast(steps=n_months)

    # Future dates
    last_date = arima_model.data.endog.index[-1]
    future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=n_months, freq='M')

    # Create DataFrames
    df_arima = pd.DataFrame({'Date': future_dates, 'ARIMA Forecast': forecast_arima}).set_index('Date')
    df_sarima = pd.DataFrame({'Date': future_dates, 'SARIMA Forecast': forecast_sarima}).set_index('Date')

    # Plot
    st.subheader("📉 Forecasted Tesla Stock Prices")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_arima, label='ARIMA Forecast', color='blue')
    ax.plot(df_sarima, label='SARIMA Forecast', color='red')
    ax.set_title(f"Forecast for Next {n_months} Months")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Show data
    st.subheader("🔍 Forecast Table")
    st.dataframe(pd.concat([df_arima, df_sarima], axis=1).round(2))
