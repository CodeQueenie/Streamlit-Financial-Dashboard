# -*- coding: utf-8 -*-
"""
Source REPO: https://github.com/hinashussain/Streamlit-Financial-Dashboard

Updated by: Nicole LeGuern

"""


import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as gostreamlit
from plotly.subplots import make_subplots
import mplfinance as mpf
from chart_module import display_chart

#import pyfolio as pf

#==============================================================================
# Helper Function to Get S&P 500 Tickers
#==============================================================================

def get_sp500_tickers():
    """Fetches the latest S&P 500 ticker symbols from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]  # Extract first table
    return table['Symbol'].tolist()

#==============================================================================
# Tab 1 Summary
#==============================================================================

from datetime import datetime, timedelta

@st.cache_data(show_spinner=True, ttl=300)
def get_stock_data(ticker, start_date=None):
    """Fetches stock data from Yahoo Finance for the given ticker symbol, starting from a specific date if provided."""
    try:
        # Calculate start date if not provided
        if not start_date:
            start_date = datetime.now() - timedelta(days=5*365.25)  # Approximating leap years
            start_date = start_date.strftime('%Y-%m-%d')  # Formatting to string for yfinance
        
        data = yf.download(ticker, start=start_date)
        
        if data.empty:
            st.warning("No data available for the selected ticker.")
        return data
    except Exception as e:
        st.error(f"An error occurred while fetching data: {str(e)}")
        return pd.DataFrame()

def tab1():
    """
    Displays the Summary tab in the Streamlit application, focusing on the last 5 years of stock data.
    """
    st.title("Summary")
    st.write("Select ticker on the left to begin")
    st.write(f"Selected Ticker: {ticker}")

    def get_stock_summary(ticker):
        """
        Retrieves and displays summary information for the given ticker symbol.
        """
        stock = yf.Ticker(ticker)
        return stock.info

    col1, col2 = st.columns([1, 1.2])  # Adjusted column width for better spacing

    with col1:
        if ticker != "-":
            summary = get_stock_summary(ticker)

            # 🏢 Display Company Overview
            st.subheader("🏢 Company Overview")
            st.markdown(f"""
                - **Company Name:** {summary.get('longName', 'N/A')}  
                - **Industry:** {summary.get('industry', 'N/A')}  
                - **Sector:** {summary.get('sector', 'N/A')}  
                - **Headquarters:** {summary.get('address1', 'N/A')}, {summary.get('city', 'N/A')}, {summary.get('country', 'N/A')}  
                - **Website:** [{summary.get('website', 'N/A')}]({summary.get('website', '#')})
            """)

            # 📝 Display Business Summary
            st.subheader("📝 Business Summary")
            st.write(summary.get("longBusinessSummary", "No summary available."))

            # 💰 Display Key Financial Metrics
            st.subheader("💰 Key Financial Metrics")
            st.markdown(f"""
                - **52-Week Range:** <span style='font-size:16px; font-weight:bold;'>{summary.get('fiftyTwoWeekLow', 'N/A')} - {summary.get('fiftyTwoWeekHigh', 'N/A')}</span>  
                - **Revenue:** ${summary.get('totalRevenue', 'N/A'):,}  
                - **Net Income:** ${summary.get('netIncomeToCommon', 'N/A'):,}  
                - **EPS:** ${summary.get('trailingEps', 'N/A')}  
                - **P/E Ratio:** {summary.get('trailingPE', 'N/A'):.2f}  
                - **Dividend Yield:** {summary.get('dividendYield', 0) * 100:.2f}%  
                - **ROA:** {summary.get('returnOnAssets', 0) * 100:.2f}%  
                - **ROE:** {summary.get('returnOnEquity', 0) * 100:.2f}%
            """, unsafe_allow_html=True)

    with col2:
        if ticker != "-":
            # 📊 Display (5Y) Stock Chart FIRST
            chartdata = get_stock_data(ticker)
            if not chartdata.empty:
                # ✅ Fix MultiIndex Issue
                if isinstance(chartdata.columns, pd.MultiIndex):
                    column_key = ('Close', ticker) if ('Close', ticker) in chartdata.columns else None
                    if column_key:
                        y_data = chartdata[column_key]  # Correctly reference column
                else:
                    y_data = chartdata["Close"] if "Close" in chartdata.columns else None

                if y_data is not None:
                    fig = px.area(chartdata, x=chartdata.index, y=y_data)

                    fig.update_layout(
                        title="📊 5-Year Closing Prices",
                        title_x=0.20,  # Centers title properly
                        title_y=0.90,  # Moves title down slightly for balance
                        title_font=dict(size=25),  # Increased size for clarity
                        margin=dict(t=80, l=50, r=50, b=50),  # Fixes modebar spacing & improves balance
                        xaxis_title="",  # Keeps year labels, removes "Date"
                        yaxis=dict(title="Stock Price", tickmode="linear", dtick=50)  # ✅ Aligns Y-axis scale
                    )

                    # ✅ Modebar Above the Title
                    st.plotly_chart(fig, use_container_width=True, config={
                        "displayModeBar": True,
                        "modeBarButtonsToAdd": ["zoom", "pan", "resetScale"],
                        "displaylogo": False
                    })
                else:
                    st.error("No valid closing price data available.")

            # 🚀 Add Extra Spacing Before Stock Performance Section
            st.markdown("<br><br>", unsafe_allow_html=True)  # Adds vertical space

            # 📈 Display Stock Performance Metrics
            st.subheader("📈 Stock Performance")
            st.markdown(f"""
                - **Current Price:** <span style='font-size:16px; font-weight:bold;'>{summary.get('currentPrice', 'N/A')}</span>  
                - **Previous Close:** <span style='font-size:16px; font-weight:bold;'>{summary.get('previousClose', 'N/A')}</span>  
                - **Day Range:** <span style='font-size:16px; font-weight:bold;'>{summary.get('dayLow', 'N/A')} - {summary.get('dayHigh', 'N/A')}</span>  
                - **Average Volume:** {summary.get('averageVolume', 'N/A'):,}  
                - **Beta (Volatility):** {summary.get('beta', 'N/A')}  
                - **Analyst Target Price:** <span style='font-size:16px; font-weight:bold;'>{summary.get('targetMeanPrice', 'N/A')}</span>  
                - **Recommendation:** {summary.get('recommendationKey', 'N/A').capitalize()}
            """, unsafe_allow_html=True)

            # 🚀 Add Extra Spacing Before Executive Team
            st.markdown("<br><br>", unsafe_allow_html=True)  # Adds vertical space

            # 👥 Display Company Executives (Now Below Stock Chart)
            st.subheader("👥 Executive Team")
            executives = summary.get("companyOfficers", [])

            if executives:
                exec_data = []
                for exec in executives:
                    exec_data.append([
                        exec.get("name", "N/A"),
                        exec.get("title", "N/A"),
                        exec.get("age", "N/A"),
                        f"${exec.get('totalPay', 0):,}" if exec.get("totalPay") else "N/A"
                    ])

                exec_df = pd.DataFrame(exec_data, columns=["Name", "Title", "Age", "Total Compensation"])
                
                # ✅ Remove Default Index Numbers
                st.dataframe(exec_df.set_index("Name"))
            else:
                st.write("No executive data available.")

#==============================================================================
# Tab 2 Chart - Stock Data Visualization
#==============================================================================


def tab2():
    """
    Displays stock price charts using mplfinance:
    1️⃣ **Line Chart** → Simple closing price trend.
    2️⃣ **Candlestick Chart** → Open-High-Low-Close (OHLC).
    """

    st.title("📈 Stock Price Chart")

    if ticker == "-":
        st.warning("⚠️ Please select a valid stock ticker.")
        return

    st.write(f"🔹 **Stock Selected:** {ticker}")

    # UI Inputs
    start_date = st.date_input("📅 Start Date", datetime.today().date() - timedelta(days=30))
    end_date = st.date_input("📅 End Date", datetime.today().date())
    chart_type = st.radio("📊 Choose Chart Type:", ["Line Chart", "Candlestick Chart"])

    # Convert dates for Yahoo Finance
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')

    @st.cache_data
    def get_stock_data(ticker, start_date, end_date):
        """Fetches stock data from Yahoo Finance."""
        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
            if df.empty:
                return None
            df.reset_index(inplace=True)

            # ✅ Flatten MultiIndex Columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

            # ✅ Rename Columns to Standard Names (Remove Ticker Prefix)
            rename_map = {col: col.split('_')[0] for col in df.columns}
            df.rename(columns=rename_map, inplace=True)

            # ✅ Convert Date Column to DatetimeIndex
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)

            return df
        except Exception as e:
            return None

    # Fetch stock data
    stock_data = get_stock_data(ticker, start_date, end_date)

    # ✅ **Simple Data Validation**
    if stock_data is None or stock_data.empty:
        st.error("❌ Unable to fetch stock data. Please try again.")
        return

    # ✅ Debugging: Show column names in case of issues
    st.write("📌 Debugging: Dataframe Columns")
    st.write(stock_data.columns.tolist())  # Show available columns

    # ✅ Check if required columns exist
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing_cols = [col for col in required_cols if col not in stock_data.columns]

    if missing_cols:
        st.error(f"⚠️ Missing columns in stock data: {missing_cols}")
        return

    # ✅ Ensure all financial columns are numeric & drop NaN rows
    stock_data[required_cols] = stock_data[required_cols].apply(pd.to_numeric, errors="coerce")
    stock_data.dropna(subset=required_cols, inplace=True)

    # ✅ Display Basic Stock Info
    latest_date = stock_data.index[-1].strftime('%Y-%m-%d')
    latest_close = stock_data["Close"].iloc[-1]

    st.subheader("📊 Stock Overview")
    st.write(f"**Closing Price on {latest_date}:** ${latest_close:,.2f}")  

    # ✅ **mplfinance Chart**
    if chart_type == "Line Chart":
        fig = mpf.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(2, 1, 1)  # Price plot
        ax2 = fig.add_subplot(2, 1, 2)  # Volume plot
        
        mpf.plot(stock_data, type='line', ax=ax1, volume=ax2, 
                style='charles', 
                volume_panel=1,
                panel_ratios=(2, 1),
                figsize=(10, 8),
                title=f'\n{ticker} Stock Price and Volume',
                tight_layout=True)
    else:  # Candlestick Chart
        fig = mpf.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(2, 1, 1)  # Price plot
        ax2 = fig.add_subplot(2, 1, 2)  # Volume plot
        
        mpf.plot(stock_data, type='candle', ax=ax1, volume=ax2,
                style='charles',
                volume_panel=1,
                panel_ratios=(2, 1),
                figsize=(10, 8),
                title=f'\n{ticker} Stock Price and Volume',
                tight_layout=True)

    # Display Plot using st.pyplot()
    st.pyplot(fig)

#==============================================================================
# Tab 3 Statistics
#==============================================================================

# Replace the si import with our adapter
from yfinance_adapter import (
    get_stats_valuation,
    get_stats,
    get_income_statement,
    get_balance_sheet,
    get_cash_flow,
    get_analysts_info
)

#The code below obtains information using get_stats_valuation and get_stats in
#Yahoo Finance. It then slices the dataframes and displays them in different 
#columns of the streamlit page under different headings.

def tab3():
    st.title("Statistics")
    st.write(ticker)
    c1, c2 = st.columns(2)
    
    with c1:
        st.header("Valuation Measures")
        if ticker != '-':
            try:
                valuation = get_stats_valuation(ticker)
                st.table(valuation)
            except Exception as e:
                st.error(f"Error fetching valuation data: {str(e)}")
                
        st.header("Financial Highlights")
        st.subheader("Fiscal Year")
         
        if ticker != '-':
            try:
                stats = get_stats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[29:31,])
            except Exception as e:
                st.error(f"Error fetching financial highlights: {str(e)}")
                
        st.subheader("Profitability")
         
        if ticker != '-':
            try:
                stats = get_stats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[31:33,])
            except Exception as e:
                st.error(f"Error fetching profitability data: {str(e)}")
                
        st.subheader("Management Effectiveness")
         
        if ticker != '-':
            try:
                stats = get_stats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[33:35,])
            except Exception as e:
                st.error(f"Error fetching management effectiveness data: {str(e)}")
         
        st.subheader("Income Statement")
         
        if ticker != '-':
            try:
                stats = get_stats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[35:43,])
            except Exception as e:
                st.error(f"Error fetching income statement data: {str(e)}")
            
        st.subheader("Balance Sheet")
         
        if ticker != '-':
            try:
                stats = get_stats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[43:49,])
            except Exception as e:
                st.error(f"Error fetching balance sheet data: {str(e)}")
         
        st.subheader("Cash Flow Statement")
         
        if ticker != '-':
            try:
                stats = get_stats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[49:,])
            except Exception as e:
                st.error(f"Error fetching cash flow statement data: {str(e)}")
                           
    with c2:
        st.header("Trading Information")
         
        st.subheader("Stock Price History")
                  
        if ticker != '-':
            try:
                stats = get_stats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[:7,])
            except Exception as e:
                st.error(f"Error fetching stock price history: {str(e)}")
         
        st.subheader("Share Statistics")
                  
        if ticker != '-':
            try:
                stats = get_stats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[7:19,])
            except Exception as e:
                st.error(f"Error fetching share statistics: {str(e)}")
         
        st.subheader("Dividends & Splits")
                  
        if ticker != '-':
            try:
                stats = get_stats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[19:29,])
            except Exception as e:
                st.error(f"Error fetching dividends & splits data: {str(e)}")
         
#==============================================================================
# Tab 4 Financials
#==============================================================================

#The code below obtains yearly and quartely financial statements from Yahoo Finance
#and displays them according the options selected by the users in streamlit. A
#combination of if statements is used to display according to the selected options.

def tab4():
    st.title("Financials")
    st.write(ticker)

    statement = st.selectbox("Show", ['Income Statement', 'Balance Sheet', 'Cash Flow'])
    period = st.selectbox("Period", ['Yearly', 'Quarterly'])

    @st.cache_data
    def get_financials(ticker, statement, period):
        yearly = period == 'Yearly'
        if statement == 'Income Statement':
            return get_income_statement(ticker, yearly=yearly)
        elif statement == 'Balance Sheet':
            return get_balance_sheet(ticker, yearly=yearly)
        elif statement == 'Cash Flow':
            return get_cash_flow(ticker, yearly=yearly)

    if ticker != '-':
        try:
            data = get_financials(ticker, statement, period)
            st.table(data)
        except Exception as e:
            st.error(f"Error fetching financial data: {str(e)}")
      
#==============================================================================
# Tab 5 Analysis
#==============================================================================

#In the code below, get_analysts_info is used to obtain the data. The output is
#in the form of a dictionary. .items() is used to get the items from the dictionary
#and then a for loop i used under which the dictionary items are changed into a list
# and each element of the list is then converted to a dataframe for displaying.

def tab5():
    st.title("Analysis")
    st.write("Currency in USD")
    st.write(ticker)

    @st.cache_data
    def get_analysis(ticker):
        analysis_dict = get_analysts_info(ticker)
        return analysis_dict.items()

    if ticker != '-':
        for i in range(6):
            analysis = get_analysis(ticker)
            df = pd.DataFrame(list(analysis)[i][1])
            st.table(df)
            
           
#==============================================================================
# Tab 6 Monte Carlo Simulation
#==============================================================================

# Performs a Monte carlo Csimulation for a specified time horizon and number of intervals.
#
# This uses Pandas.read_html to scrape the S&P 500 tickers from Wikipedia
# Then it uses yfinance to pull data for stock charts. 
#
# Key changes made:

    # For the Monte Carlo simulation (tab6):

    # Removed dependency on legendHandles
    # Simplified legend creation
    # Added better error handling for MultiIndex columns
    # Improved type conversion for price values
    # Added more descriptive error messages
    
    # For the Portfolio Trend (tab7):

    # Replaced DataFrame creation with more efficient concatenation
    # Added pre-allocation of data dictionary
    # Improved error handling for individual ticker fetches
    # Added better plot formatting
    # Added data validation checks
    
    # General improvements:

    # Added docstrings
    # Improved error messages
    # Added type hints where appropriate
    # Optimized DataFrame operations
    # Added data validation checks
    
    # These changes should resolve both the legend error and the DataFrame fragmentation warning.

# Generated by Nicole LeGuern

def tab6():
    """Monte Carlo simulation for stock price prediction."""
    st.title("Monte Carlo Simulation")
    st.write(ticker)

    simulations = st.selectbox("Number of Simulations (n)", [200, 500, 1000])
    time_horizon = st.selectbox("Time Horizon (t)", [30, 60, 90])

    @st.cache_data
    def montecarlo(ticker, time_horizon, simulations):
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=30)
            stock_price = yf.download(ticker, start=start_date, end=end_date)
            
            # Handle MultiIndex columns if present
            if isinstance(stock_price.columns, pd.MultiIndex):
                close_price = stock_price[('Close', ticker)]
            else:
                close_price = stock_price['Close']
                
            daily_return = close_price.pct_change()
            daily_volatility = np.std(daily_return)

            # Pre-allocate DataFrame with known size
            simulation_df = pd.DataFrame(index=range(time_horizon), columns=range(simulations))
            last_price = float(close_price.iloc[-1])

            for i in range(simulations):
                next_price = last_price
                for j in range(time_horizon):
                    future_return = np.random.normal(0, daily_volatility)
                    next_price = next_price * (1 + future_return)
                    simulation_df.iloc[j, i] = next_price

            return simulation_df, last_price
        except Exception as e:
            st.error(f"Error in Monte Carlo simulation: {str(e)}")
            return None, None

    if ticker != "-":
        try:
            mc, last_price = montecarlo(ticker, time_horizon, simulations)
            if mc is not None and last_price is not None:
                fig, ax = plt.subplots(figsize=(15, 10))
                ax.plot(mc)
                plt.title(f'Monte Carlo simulation for {ticker} stock price in next {time_horizon} days')
                plt.xlabel('Day')
                plt.ylabel('Price')
                
                # Add horizontal line for current price
                current_price_line = ax.axhline(y=last_price, color='red', linestyle='-')
                
                # Create legend without using legendHandles
                ax.legend(['Simulations', f'Current stock price: ${last_price:,.2f}'])
                
                st.pyplot(fig)
            else:
                st.error("Failed to generate Monte Carlo simulation")
        except Exception as e:
            st.error(f"Error displaying Monte Carlo simulation: {str(e)}")

def tab7():
    """Portfolio trend visualization with optimized DataFrame handling."""
    st.title("Your Portfolio's Trend")
    alltickers = get_sp500_tickers()
    selected_tickers = st.multiselect("Select tickers in your portfolio", 
                                    options=alltickers, 
                                    default=[alltickers[0]])

    if selected_tickers:
        try:
            # Fetch all data at once and combine efficiently
            data_dict = {}
            for ticker in selected_tickers:
                try:
                    data = yf.download(ticker, period="5Y")['Close']
                    data_dict[ticker] = data
                except Exception as e:
                    st.warning(f"Error fetching data for {ticker}: {str(e)}")
            
            # Combine all series at once using concat
            df = pd.concat(data_dict, axis=1)
            
            # Create and display plot
            fig = px.line(df, title="Portfolio Trend (5 Years)")
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Price",
                showlegend=True
            )
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"Error creating portfolio visualization: {str(e)}")
    else:
        st.warning("Please select at least one ticker.")

# Generated by Nicole LeGuern

#==============================================================================
# Tab 7 Your Portfolio's Trend
#==============================================================================

#The code below uses a multiselect box to allow user to select multiple tickers.
#Then a new dataframe is created with each ticker as a column. A for loop is used to
#populate each column with the close price of that ticker. Then plotly is used to 
#visualize the trend of the selected portfolio
#Reference:
#https://blog.quantinsti.com/stock-market-data-analysis-python/


def tab7():
    # This uses Pandas.read_html to scrape the S&P 500 tickers from Wikipedia
    # Then it uses yfinance to pull data for stock charts. 
    st.title("Your Portfolio's Trend")
    alltickers = get_sp500_tickers()
    selected_tickers = st.multiselect("Select tickers in your portfolio", options=alltickers, default=[alltickers[0]])
    # add debugging or print statement
    
    df = pd.DataFrame(columns=selected_tickers)
    for ticker in selected_tickers:
        df[ticker] = yf.download(ticker, period="5Y")['Close']
    
    fig = px.line(df)
    st.plotly_chart(fig)
    
#==============================================================================
# Main body
#==============================================================================

def run():
    ticker_list = ['-'] + get_sp500_tickers()
    global ticker
    ticker = st.sidebar.selectbox("Select a ticker", ticker_list)
    select_tab = st.sidebar.radio("Select tab", [
        'Summary',
        'Chart',
        'Statistics',
        'Financials',
        'Analysis',
        'Monte Carlo Simulation',
        "Your Portfolio's Trend"
    ])
    
    if select_tab == 'Summary':
        tab1()
    elif select_tab == 'Chart':
        tab2()
    elif select_tab == 'Statistics':
        tab3()
    elif select_tab == 'Financials':
        tab4()
    elif select_tab == 'Analysis':
        tab5()
    elif select_tab == 'Monte Carlo Simulation':
        tab6()
    elif select_tab == "Your Portfolio's Trend":
        tab7()

if __name__ == "__main__":
    run()
