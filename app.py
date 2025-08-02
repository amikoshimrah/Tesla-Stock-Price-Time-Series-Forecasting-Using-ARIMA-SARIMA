import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Tesla Forecasting", layout="wide")

# Load models along with their training end date
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

models = load_models()

# Sidebar inputs
with st.sidebar:
    st.header("📊 Forecast Settings")
    model_choice = st.selectbox("Select Model", list(models.keys()))
    forecast_months = st.slider("Forecast Period (months)", 1, 36, 12)
    st.write("🔵 ARIMA: Suitable for short-term trends")
    st.write("🔴 SARIMA: Captures seasonal effects")

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

    # Plot forecast
    st.subheader("📉 Forecasted Tesla Stock Prices")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(forecast_df, label=f'{model_choice} Forecast', color='blue' if model_choice == "ARIMA" else 'red')
    ax.set_title(f"{model_choice} Forecast for Next {forecast_months} Months")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # Show table
    st.subheader("🔍 Forecast Table")
    st.dataframe(forecast_df.round(2))
