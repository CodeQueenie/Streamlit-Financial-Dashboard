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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
                - **Market Cap:** ${summary.get('marketCap', 'N/A'):,}  
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
# Tab 2 Chart
#==============================================================================


#The code below divides the streamlit page into 5 columns. The first two columns
#have a date picker option to select start and end dates and the the other three
#have dropdown selection boxes for duration, interval, and type of plot.

def tab2():
    st.title("Chart")
    st.write(ticker)
    
    st.write("Set duration to '-' to select date range")
    
    c1, c2, c3, c4,c5 = st.columns((1,1,1,1,1))
    
    with c1:
        
        start_date = st.date_input("Start date", datetime.today().date() - timedelta(days=30))
        
    with c2:
        
        end_date = st.date_input("End date", datetime.today().date())        
        
    with c3:
        
        duration = st.selectbox("Select duration", ['-', '1Mo', '3Mo', '6Mo', 'YTD','1Y', '3Y','5Y', 'MAX'])          
        
    with c4: 
        
        inter = st.selectbox("Select interval", ['1d', '1mo'])
        
    with c5:
        
        plot = st.selectbox("Select Plot", ['Line', 'Candle'])
        
 
#The code below first obtains all the data using the download option from yahoo finance.
#It then creates a column for the simple moving average, makes the date index into a column
#and then subsets the dataframe to get just the date and and SMA column.
#Then if a duration is selected from the dropdown, data for that duration is downloaded
# and the SMA column is merged to the dataframe. If a duration is not selected then
#automatically the specified date range is used to get the data and that is also merged
#with the SMA column
#References:
#https://towardsdatascience.com/data-science-in-finance-56a4d99279f7

           
             
    @st.cache_data              
    def getchartdata(ticker):
        SMA = yf.download(ticker, period = 'MAX')
        SMA['SMA'] = SMA['Close'].rolling(50).mean()
        SMA = SMA.reset_index()
        SMA = SMA[['Date', 'SMA']]
        
        if duration != '-':        
            chartdata1 = yf.download(ticker, period = duration, interval = inter)
            chartdata1 = chartdata1.reset_index()
            chartdata1 = chartdata1.merge(SMA, on='Date', how='left')
            return chartdata1
        else:
            chartdata2 = yf.download(ticker, start_date, end_date, interval = inter)
            chartdata2 = chartdata2.reset_index()
            chartdata2 = chartdata2.merge(SMA, on='Date', how='left')                             
            return chartdata2
    
#The code below uses plotly to visualize the data. Subplots from plotly is used to make 2 y axis.
#First y axis shows the stock close price and SMA and the second is used to show volume. 
#Plotly graph objects are used to add graphs to the axes.The range for the y axis for 
#volume is manipulated so that the bars appear small.
#References:
#https://plotly.com/python/multiple-axes/   
#https://plotly.com/python/candlestick-charts/        
        
    if ticker != '-':
            chartdata = getchartdata(ticker) 
            
                       
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            if plot == 'Line':
                fig.add_trace(go.Scatter(x=chartdata['Date'], y=chartdata['Close'], mode='lines', 
                                         name = 'Close'), secondary_y = False)
            else:
                fig.add_trace(go.Candlestick(x = chartdata['Date'], open = chartdata['Open'], 
                                             high = chartdata['High'], low = chartdata['Low'], close = chartdata['Close'], name = 'Candle'))
              
                    
            fig.add_trace(go.Scatter(x=chartdata['Date'], y=chartdata['SMA'], mode='lines', name = '50-day SMA'), secondary_y = False)
            
            fig.add_trace(go.Bar(x = chartdata['Date'], y = chartdata['Volume'], name = 'Volume'), secondary_y = True)

            fig.update_yaxes(range=[0, chartdata['Volume'].max()*3], showticklabels=False, secondary_y=True)
        
      
            st.plotly_chart(fig)
           
             

#==============================================================================
# Tab 3 Statistics
#==============================================================================

#The code below obtains information using get_stats_valuation and get_stats in
#Yahoo Finance. It then slices the dataframes and displays them in different 
#columns of the streamlit page under different headings.

def tab3():
     st.title("Statistics")
     st.write(ticker)
     c1, c2 = st.columns(2)
     
         
     
     with c1:
         st.header("Valuation Measures")
         #@st.cache
         def getvaluation(ticker):
                 return si.get_stats_valuation(ticker)
    
         if ticker != '-':
                valuation = getvaluation(ticker)
                valuation[1] = valuation[1].astype(str)
                valuation = valuation.rename(columns = {0: 'Attribute', 1: ''})
                valuation.set_index('Attribute', inplace=True)
                st.table(valuation)
                
        
         st.header("Financial Highlights")
         st.subheader("Fiscal Year")
         
         #@st.cache
         def getstats(ticker):
                 return si.get_stats(ticker)
         
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[29:31,])
                
        
         st.subheader("Profitability")
         
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[31:33,])
                
                
                
         st.subheader("Management Effectiveness")
         
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[33:35,])
         
         
                
         st.subheader("Income Statement")
         
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[35:43,])  
            
         
         st.subheader("Balance Sheet")
         
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[43:49,])
         
         st.subheader("Cash Flow Statement")
         
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[49:,])
         
        
                           
     with c2:
         st.header("Trading Information")
         
         
         st.subheader("Stock Price History")
                  
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[:7,])
         
         st.subheader("Share Statistics")
                  
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[7:19,])
         
         st.subheader("Dividends & Splits")
                  
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[19:29,])
         
         
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
        if statement == 'Income Statement':
            return si.get_income_statement(ticker, yearly=(period == 'Yearly'))
        elif statement == 'Balance Sheet':
            return si.get_balance_sheet(ticker, yearly=(period == 'Yearly'))
        elif statement == 'Cash Flow':
            return si.get_cash_flow(ticker, yearly=(period == 'Yearly'))

    if ticker != '-':
        data = get_financials(ticker, statement, period)
        st.table(data)   
      
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
        analysis_dict = si.get_analysts_info(ticker)
        return analysis_dict.items()

    if ticker != '-':
        for i in range(6):
            analysis = get_analysis(ticker)
            df = pd.DataFrame(list(analysis)[i][1])
            st.table(df)
            
           
#==============================================================================
# Tab 6 Monte Carlo Simulation
#==============================================================================

#The code below performs and displays the monte carlo simulation for a specified
#time horizon and number of intervals

def tab6():
    st.title("Monte Carlo Simulation")
    st.write(ticker)

    simulations = st.selectbox("Number of Simulations (n)", [200, 500, 1000])
    time_horizon = st.selectbox("Time Horizon (t)", [30, 60, 90])

    @st.cache_data
    def montecarlo(ticker, time_horizon, simulations):
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30)
        stock_price = yf.download(ticker, start=start_date, end=end_date)
        close_price = stock_price['Close']
        daily_return = close_price.pct_change()
        daily_volatility = np.std(daily_return)

        simulation_df = pd.DataFrame()
        for i in range(simulations):
            next_price = []
            last_price = close_price.iloc[-1]
            for x in range(time_horizon):
                future_return = np.random.normal(0, daily_volatility)
                future_price = last_price * (1 + future_return)
                next_price.append(future_price)
                last_price = future_price
            simulation_df[i] = next_price
        return simulation_df

    if ticker != "-":
        mc = montecarlo(ticker, time_horizon, simulations)
        close_price = yf.download(ticker, period="30d")['Close']
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(mc)
        plt.title(f'Monte Carlo simulation for {ticker} stock price in next {time_horizon} days')
        plt.xlabel('Day')
        plt.ylabel('Price')
        plt.axhline(y=close_price.iloc[-1], color='red')
        plt.legend([f'Current stock price: {np.round(close_price.iloc[-1], 2)}'])
        ax.get_legend().legendHandles[0].set_color('red')
        st.pyplot(fig)

    
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
