# Streamlit Financial Dashboard

A real-time financial dashboard built with Streamlit and Yahoo Finance API, providing interactive stock analysis and visualization tools.

> **Note**: This is an expanded version of the original [Streamlit Financial Dashboard](https://github.com/hinashussain/Streamlit-Financial-Dashboard) by Hina S. Hussain, with significant modifications and improvements.

## Features

- Real-time stock data tracking
- Interactive 5-year historical price charts
- Company overview and metrics
- Financial data analysis
- Executive team information
- Improved UI and performance

## Installation

1. Create a virtual environment:
```bash
# Using Conda (Recommended)
conda create --name streamlit_app python=3.9
conda activate streamlit_app

# Or using venv
python -m venv streamlit_env
source streamlit_env/bin/activate  # Unix/MacOS
streamlit_env\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Dependencies

- streamlit
- numpy
- pandas
- matplotlib
- yfinance
- plotly

## Usage

1. Launch the application
2. Enter a stock ticker in the sidebar (e.g., AAPL, TSLA, GOOGL)
3. Explore financial metrics, stock trends, and company information
4. Interact with charts using zoom and pan features

## Roadmap

- [ ] Add technical indicators (moving averages, RSI, MACD)
- [ ] Implement portfolio tracking & watchlists
- [ ] Optimize performance
- [ ] Integrate additional data sources

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original template by [Hina S. Hussain](https://github.com/hinashussain/Streamlit-Financial-Dashboard)
- Modified and enhanced by Nicole LeGuern

