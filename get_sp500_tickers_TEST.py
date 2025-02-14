

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


get_sp500_tickers()

# url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# table = pd.read_html(url)[0]  # Extract first table
# table['Symbol'].tolist()
