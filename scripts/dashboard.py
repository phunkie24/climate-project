#!/usr/bin/env python3
"""
Interactive Streamlit dashboard for climate rainfall analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_climate_data
from src.models import load_predictor


# Page config
st.set_page_config(
    page_title="Climate Rainfall Analysis",
    page_icon="🌧️",
    layout="wide"
)


@st.cache_data
def load_data():
    """Load processed data"""
    try:
        return load_climate_data('data/processed/climate_data_featured.csv')
    except:
        st.error("Data not found! Please run: python scripts/process_data.py")
        return None


def main():
    """Main dashboard application"""
    
    # Title
    st.title("🌧️ Climate Rainfall Analysis Dashboard")
    st.markdown("### Sub-Saharan Africa (1991-2023)")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Sidebar filters
    st.sidebar.header("📊 Filters")
    
    # Region filter
    regions = ['All'] + sorted(df['region'].dropna().unique().tolist())
    selected_region = st.sidebar.selectbox("Select Region", regions)
    
    # Country filter
    if selected_region != 'All':
        countries = df[df['region'] == selected_region]['country'].unique()
    else:
        countries = df['country'].unique()
    
    selected_countries = st.sidebar.multiselect(
        "Select Countries",
        options=sorted(countries),
        default=[]
    )
    
    # Year range
    year_range = st.sidebar.slider(
        "Year Range",
        min_value=int(df['year'].min()),
        max_value=int(df['year'].max()),
        value=(1991, 2023)
    )
    
    # Filter data
    df_filtered = df[
        (df['year'] >= year_range[0]) &
        (df['year'] <= year_range[1])
    ]
    
    if selected_region != 'All':
        df_filtered = df_filtered[df_filtered['region'] == selected_region]
    
    if selected_countries:
        df_filtered = df_filtered[df_filtered['country'].isin(selected_countries)]
    
    # Overview metrics
    st.header("📈 Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Countries",
            len(df_filtered['country'].unique())
        )
    
    with col2:
        avg_precip = df_filtered['precipitation(mm)'].mean()
        st.metric(
            "Avg Rainfall",
            f"{avg_precip:.1f} mm"
        )
    
    with col3:
        # Calculate trend
        yearly = df_filtered.groupby('year')['precipitation(mm)'].mean()
        if len(yearly) > 1:
            trend = np.polyfit(yearly.index, yearly.values, 1)[0]
            st.metric(
                "Trend",
                f"{trend:.2f} mm/yr",
                delta=f"{trend*10:.1f} mm/decade"
            )
        else:
            st.metric("Trend", "N/A")
    
    with col4:
        years_covered = year_range[1] - year_range[0] + 1
        st.metric(
            "Years",
            years_covered
        )
    
    # Temporal trends
    st.header("📊 Temporal Trends")
    
    tab1, tab2 = st.tabs(["Yearly Trends", "Regional Comparison"])
    
    with tab1:
        if selected_countries:
            # Country-specific trends
            fig = go.Figure()
            
            for country in selected_countries:
                country_data = df_filtered[df_filtered['country'] == country]
                yearly = country_data.groupby('year')['precipitation(mm)'].mean()
                
                fig.add_trace(go.Scatter(
                    x=yearly.index,
                    y=yearly.values,
                    mode='lines+markers',
                    name=country
                ))
            
            fig.update_layout(
                title="Rainfall Trends by Country",
                xaxis_title="Year",
                yaxis_title="Precipitation (mm)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Overall trend
            yearly = df_filtered.groupby('year')['precipitation(mm)'].mean()
            
            fig = px.line(
                x=yearly.index,
                y=yearly.values,
                labels={'x': 'Year', 'y': 'Precipitation (mm)'},
                title='Average Rainfall Over Time'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Regional comparison
        regional = df_filtered.groupby(['year', 'region'])['precipitation(mm)'].mean().reset_index()
        
        fig = px.line(
            regional,
            x='year',
            y='precipitation(mm)',
            color='region',
            title='Regional Rainfall Trends',
            labels={'precipitation(mm)': 'Precipitation (mm)', 'year': 'Year'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Spatial analysis
    st.header("🗺️ Spatial Patterns")
    
    # Country ranking
    country_avg = df_filtered.groupby('country')['precipitation(mm)'].mean().sort_values(ascending=False)
    
    fig = px.bar(
        x=country_avg.values[:20],
        y=country_avg.index[:20],
        orientation='h',
        labels={'x': 'Average Precipitation (mm)', 'y': 'Country'},
        title='Top 20 Countries by Rainfall'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction tool
    st.header("🤖 Rainfall Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pred_country = st.selectbox("Country", sorted(df['country'].unique()))
        pred_year = st.number_input("Year", min_value=2024, max_value=2050, value=2025)
        pred_temp = st.slider("Average Temperature (°C)", 15.0, 35.0, 25.0, 0.5)
    
    with col2:
        pred_humidity = st.slider("Average Humidity (%)", 30.0, 90.0, 60.0, 1.0)
        pred_co2 = st.slider("Atmospheric CO₂ (ppm)", 400.0, 550.0, 420.0, 5.0)
        pred_cloud = st.slider("Cloud Cover (%)", 20.0, 80.0, 50.0, 5.0)
    
    if st.button("🔮 Predict Rainfall"):
        try:
            predictor = load_predictor('xgboost')
            
            # Simple prediction (may need more features for actual model)
            st.info("Note: Prediction requires trained model with all features")
            
            # Show historical average as reference
            historical = df[df['country'] == pred_country]['precipitation(mm)'].mean()
            st.success(f"Historical average for {pred_country}: **{historical:.1f} mm**")
            
        except Exception as e:
            st.warning(f"Model not available: {str(e)}")
            st.info("Please train model first: python scripts/train_model.py")
    
    # Data table
    st.header("📋 Data Explorer")
    
    # Show filtered data
    st.dataframe(
        df_filtered[['country', 'year', 'region', 'precipitation(mm)', 'avg_temp_c']].head(100),
        use_container_width=True
    )
    
    # Download button
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data",
        data=csv,
        file_name='climate_data_filtered.csv',
        mime='text/csv'
    )
    
    # Footer
    st.markdown("---")
    st.markdown("**Data Source**: CHIRPS, ERA5, NOAA | **Period**: 1991-2023 | **Countries**: 49")


if __name__ == '__main__':
    main()
