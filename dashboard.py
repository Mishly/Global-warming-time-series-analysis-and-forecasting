import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import io

st.title('Temperature Forecasting')

# Upload CSV File
uploaded_file = st.file_uploader("Upload your temperature CSV file", type=["csv"])

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)

    # Preview
    st.subheader("Preview of Uploaded Data")
    st.write(df.head())

    # Ensure datetime index
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)

    if 'DAYTON_MW' not in df.columns:
        st.error("Uploaded CSV must contain a 'DAYTON_MW' column.")
    else:
        # Forecasting functions
        def forecast_arima(data, steps):
            model = ARIMA(data['DAYTON_MW'], order=(5, 1, 0))
            model_fit = model.fit()
            forecast = model_fit.get_forecast(steps)
            return forecast.predicted_mean, forecast.conf_int()

        def forecast_sarima(data, steps):
            model = SARIMAX(data['DAYTON_MW'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            model_fit = model.fit()
            forecast = model_fit.get_forecast(steps)
            return forecast.predicted_mean, forecast.conf_int()

        def moving_average(data, window_size, steps):
            last_temp_ma = data['DAYTON_MW'].rolling(window=window_size).mean().iloc[-1]
            forecast = [last_temp_ma] * steps
            return pd.Series(forecast, index=pd.date_range(start=data.index[-1], periods=steps+1, freq='D')[1:]), None

        # Sidebar inputs
        st.sidebar.header('User Input Parameters')
        model_type = st.sidebar.selectbox('Select Model:', ('ARIMA', 'SARIMA', 'Moving Average'))
        forecast_steps = st.sidebar.number_input('Forecast Steps:', min_value=1, max_value=30)

        if model_type == 'Moving Average':
            window_size = st.sidebar.number_input('Window Size:', min_value=1)
        else:
            window_size = None

        if st.button('Run Forecast'):
            if model_type == 'ARIMA':
                forecast_result, conf_int = forecast_arima(df[-365:], forecast_steps)
                st.write(f"ARIMA Forecast for next {forecast_steps} steps:")
            elif model_type == 'SARIMA':
                forecast_result, conf_int = forecast_sarima(df[-365:], forecast_steps)
                st.write(f"SARIMA Forecast for next {forecast_steps} steps:")
            else:
                forecast_result, conf_int = moving_average(df[-365:], window_size, forecast_steps)
                st.write(f"Moving Average Forecast for next {forecast_steps} steps:")

            # Plotting
            st.subheader("Forecast Plot")
            fig, ax = plt.subplots(figsize=(10, 4))
            df['DAYTON_MW'].iloc[-100:].plot(ax=ax, label='Historical')
            forecast_result.plot(ax=ax, label='Forecast', color='red')

            # Confidence intervals
            if conf_int is not None:
                ax.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3, label='Confidence Interval')

            ax.legend()
            ax.set_title("Forecast with Confidence Intervals")
            st.pyplot(fig)

            # Download forecast
            st.subheader("Download Forecast")
            csv = forecast_result.to_csv(index=True)
            st.download_button(
                label="Download forecast as CSV",
                data=csv,
                file_name='temperature_forecast.csv',
                mime='text/csv',
            )

else:
    st.info("Please upload a CSV file containing temperature data.")

