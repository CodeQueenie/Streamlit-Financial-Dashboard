# Streamlit Financial Dashboard (WIP)

This is a **revamped and expanded** version of the original **Streamlit Financial Dashboard** by [Hina S. Hussain](https://github.com/hinashussain/Streamlit-Financial-Dashboard). While I used the original project as a **template**, I have **significantly modified and rebuilt** many sections to improve usability, design, and performance.

This is still a **work in progress (WIP)** as I continue making improvements.

---

## 🚀 About This Project
This **interactive financial dashboard** allows users to track **real-time stock data**, visualize **historical trends**, and analyze **financial metrics** using **Streamlit** and **Yahoo Finance API**.

**Originally a template, this version has been heavily modified** with an improved UI, better data formatting, and optimized performance.

---

## 🔥 What's New? (Modifications)
- **Better UI & Styling** – Improved layout, colors, spacing, and chart positioning.
- **Fixed Chart Issues** – The **5-Year Closing Price chart** is now **properly formatted & aligned**.
- **Enhanced Financial Data Display** – Cleaner number formatting, better spacing, and readability.
- **Executive Team Section** – Table now looks cleaner, without unnecessary index numbers.
- **Backend Performance Fixes** – Resolved Yahoo Finance API **MultiIndex issues**, improved data handling.
- **Added Conda & Pip Installation Guide** – Setup instructions now include **virtual environments**.
- **Work in Progress (WIP)** – Continuing to add new features, fix bugs, and improve the app.

---

## ⚡ Features
- **Real-time Stock Data** – Get **live stock prices, financial metrics, and market cap**.
- **Historical Stock Charts** – View **5-year trends** with zoom & pan capabilities.
- **Company Overview** – See key **company details like industry, sector, and location**.
- **Financial Metrics** – Track **market cap, revenue, EPS, ROE, and more**.
- **Executive Team Data** – View **CEO, CFO, and other leadership info** in a table.

---

## 🛠 Installation & Setup

### 1️⃣ **Set Up a Virtual Environment**  
Using **Conda** (Recommended):
```bash
conda create --name streamlit_app python=3.9
conda activate streamlit_app

Using Virtualenv:

bash
Copy
Edit
python -m venv streamlit_env
source streamlit_env/bin/activate  # (Mac/Linux)
streamlit_env\Scripts\activate  # (Windows)
2️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
OR manually install packages:

bash
Copy
Edit
pip install streamlit yahoo_fin numpy pandas matplotlib yfinance plotly
3️⃣ Run the Streamlit App
bash
Copy
Edit
streamlit run app.py
📚 Packages Used
These packages are used in the app:

python
Copy
Edit
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
💡 pyfolio is currently commented out but may be used in future updates.

🎯 How to Use
Enter a stock ticker in the sidebar (e.g., AAPL, TSLA, GOOGL).
View financial metrics, stock trends, and company details.
Interact with stock charts – zoom, pan, and analyze trends.
Explore company summaries – Learn about industry, sector, and market position.
