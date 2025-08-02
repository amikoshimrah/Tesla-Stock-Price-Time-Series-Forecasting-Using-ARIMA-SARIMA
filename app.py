import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Tesla Forecasting", layout="wide")

# Load models
@st.cache_resource
def load_models():
    with open("arima_model_tsa.pkl", "rb") as f_arima:
        arima_model = pickle.load(f_arima)
    with open("sarima_model_tsa.pkl", "rb") as f_sarima:
        sarima_model = pickle.load(f_sarima)
    return {"ARIMA": arima_model, "SARIMA": sarima_model}

models = load_models()

# --------------------
# Sidebar inputs
# --------------------
with st.sidebar:
    st.header("📊 Forecast Settings")
    model_choice = st.selectbox("Select Model", list(models.keys()))
    forecast_months = st.slider("Forecast Period (months)", 1, 36, 12)
    st.write("🔵 ARIMA: Suitable for short-term trends")
    st.write("🔴 SARIMA: Captures seasonal effects")

# --------------------
# Title
# --------------------
st.title("📈 Tesla Stock Price Forecasting")
st.markdown(f"Forecasting with **{model_choice}** model")

# --------------------
# Generate forecast
# --------------------
if st.button("🔮 Generate Forecast"):
    model = models[model_choice]
    
    # Forecast future prices
    forecast = model.forecast(steps=forecast_months)
    
    # Estimate future dates
    last_date = model.data.endog.index[-1]
    future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=forecast_months, freq='M')
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast}).set_index('Date')

    # --------------------
    # Plot forecast
    # --------------------
    st.subheader("📉 Forecasted Tesla Stock Prices")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(forecast_df, label=f'{model_choice} Forecast', color='blue' if model_choice == "ARIMA" else 'red')
    ax.set_title(f"{model_choice} Forecast for Next {forecast_months} Months")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # --------------------
    # Show table
    # --------------------
    st.subheader("🔍 Forecast Table")
    st.dataframe(forecast_df.round(2))
