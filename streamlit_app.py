import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="CO₂ Forecast", layout="centered")
st.title("🌍 Country-wise CO₂ Emissions Forecast")

uploaded_file = st.file_uploader("📁 Upload your CSV file (e.g., co2_emissions_kt_by_country.csv)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df[df['year'] >= 2000]
        df['value_mt'] = df['value'] / 1000.0
        df_clean = df[['country_name', 'year', 'value_mt']].dropna()

        countries = sorted(df_clean['country_name'].unique())
        country = st.selectbox("🌍 Select a Country", countries)

        col1, col2 = st.columns(2)
        with col1:
            start_year = st.number_input("📅 Start Year", min_value=2024, max_value=2100, value=2025, step=1)
        with col2:
            end_year = st.number_input("📅 End Year", min_value=2025, max_value=2100, value=2040, step=1)

        if start_year >= end_year:
            st.error("❌ End year must be greater than start year.")
        else:
            df_country = df_clean[df_clean['country_name'] == country]
            df_grouped = df_country.groupby('year')['value_mt'].sum().reset_index()
            df_grouped.columns = ['Year', 'CO2 Emissions (Mt)']
            df_grouped['Type'] = 'Historical'

            X = df_grouped['Year'].values.reshape(-1, 1)
            y = df_grouped['CO2 Emissions (Mt)'].values

            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)

            model = LinearRegression()
            model.fit(X_poly, y)

            future_years = np.arange(start_year, end_year + 1).reshape(-1, 1)
            future_X_poly = poly.transform(future_years)
            predictions = model.predict(future_X_poly)

            df_forecast = pd.DataFrame({
                'Year': future_years.flatten(),
                'CO2 Emissions (Mt)': predictions,
                'Type': 'Forecast'
            })

            df_combined = pd.concat([df_grouped, df_forecast])

            st.subheader("📊 Historical Data")
            st.dataframe(df_grouped)

            st.subheader(f"📈 Forecasted CO₂ Emissions ({start_year}–{end_year})")
            st.dataframe(df_forecast)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_grouped['Year'], y=df_grouped['CO2 Emissions (Mt)'],
                                     mode='lines+markers', name='Historical'))
            fig.add_trace(go.Scatter(x=df_forecast['Year'], y=df_forecast['CO2 Emissions (Mt)'],
                                     mode='lines+markers', name='Forecast',
                                     line=dict(dash='dash')))
            fig.update_layout(title=f"{country} CO₂ Emissions Forecast",
                              xaxis_title='Year', yaxis_title='CO₂ Emissions (Mt)',
                              template='plotly_white')
            st.plotly_chart(fig)

            csv = df_combined.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Combined Historical + Forecast CSV",
                               data=csv,
                               file_name=f"{country}_forecast_{start_year}_{end_year}.csv")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")