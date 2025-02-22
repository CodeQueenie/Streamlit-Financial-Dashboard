"""
Stock Chart Module with mplfinance integration.
Fixed implementation for stock price and volume visualization.

Generated by Nicole LeGuern
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf
from datetime import datetime, timedelta

def display_chart(ticker: str, start_date: str, end_date: str, chart_type: str = "Line Chart") -> None:
    """
    Display stock price chart with volume using mplfinance.
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        chart_type (str): Type of chart ("Line Chart" or "Candlestick Chart")
    """
    try:
        # Fetch stock data
        stock_data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
        
        if stock_data.empty:
            st.error("No data available for the selected period")
            return
            
        # Convert price columns to float with enhanced error handling
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col in stock_data.columns:
                try:
                    # Handle any string or object data types
                    if stock_data[col].dtype == 'object':
                        # Remove any currency symbols or commas
                        stock_data[col] = stock_data[col].replace('[\$,]', '', regex=True)
                    # Convert to numeric, replacing errors with NaN
                    stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')
                    # Verify the conversion was successful
                    if stock_data[col].isna().all():
                        raise ValueError(f"Column {col} could not be converted to numeric values")
                except Exception as e:
                    st.error(f"Error converting {col} to numeric: {str(e)}")
                    return
        
        # Drop any rows with NaN values after conversion
        stock_data = stock_data.dropna(subset=required_cols)
        
        # Verify we still have data after cleaning
        if stock_data.empty:
            st.error("No valid data remains after cleaning")
            return
            
        # Prepare the plot style
        mc = mpf.make_marketcolors(
            up='green',
            down='red',
            edge='inherit',
            volume='in',
            wick='inherit'
        )
        s = mpf.make_mpf_style(
            marketcolors=mc,
            gridstyle=':', 
            y_on_right=False
        )
        
        # Create the plot
        kwargs = dict(
            type='candle' if chart_type == "Candlestick Chart" else 'line',
            volume=True,
            title=f'\n{ticker} Stock Price and Volume',
            figsize=(10, 8),
            style=s,
            panel_ratios=(2, 1),
            volume_panel=1
        )
        
        # Verify data types before plotting
        for col in required_cols:
            if not np.issubdtype(stock_data[col].dtype, np.number):
                raise ValueError(f"Column {col} contains non-numeric data: {stock_data[col].dtype}")
        
        fig, axlist = mpf.plot(
            stock_data,
            returnfig=True,
            **kwargs
        )
        
        # Adjust layout
        fig.tight_layout()
        
        # Display in Streamlit
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error displaying chart: {str(e)}")
        # Add detailed debugging information
        if 'stock_data' in locals():
            st.write("Data Types:")
            st.write(stock_data.dtypes)
            st.write("Data Sample:")
            st.write(stock_data.head())
            st.write("Column Statistics:")
            st.write(stock_data.describe())

def main():
    """Main function for testing the chart module."""
    st.title("📈 Stock Price Chart")
    
    # Test with a sample ticker
    ticker = "AAPL"
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=30)
    
    display_chart(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

if __name__ == "__main__":
    main()

# Generated by Nicole LeGuern