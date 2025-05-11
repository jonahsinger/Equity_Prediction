import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# list of 20 small cap that have been around for 25 years
large_cap_tickers = [
    'AAPL', 'MSFT', 'JNJ', 'PG', 'KO', 'PEP', 'WMT', 'HD', 'XOM', 'CVX',
    'JPM', 'BAC', 'WFC', 'MRK', 'PFE', 'INTC', 'CSCO', 'VZ', 'T', 'DIS'
]

# list of 20 mid cap that have been around for 25 years
mid_cap_tickers = [
    'AAP', 'NUE', 'PKI', 'ROL', 'SJM', 'HRB', 'HSY', 'CLX', 'CPB', 'MKC',
    'WSM', 'FLO', 'LEG', 'HAS', 'PII', 'SWK', 'THO', 'MDC', 'TUP', 'SEE'
]

# list of 20 small cap that have been around for 25 year
small_cap_tickers = [
    'BGS', 'CALM', 'CATO', 'CBZ', 'CVGW', 'FIZZ', 'HELE', 'LANC', 'MATW', 'PCYG',
    'PNM', 'PRGS', 'SLP', 'TILE', 'TREX', 'UEIC', 'UVV', 'WINA', 'WDFC', 'SCX'
]

end_date = datetime.now()
start_date = end_date - timedelta(days=25*365)

def fetch_stock_data(tickers, start_date, end_date):
    print(f"Fetching price and volume data for {len(tickers)} stocks...")
    
    extended_start = end_date - timedelta(days=25*365 + 30)

    price_data = yf.download(
        tickers, 
        start=extended_start, 
        end=end_date,
        group_by='ticker',
        auto_adjust=True
    )
    
    if len(tickers) > 1:
        close_prices = pd.DataFrame({ticker: price_data[ticker]['Close'] for ticker in tickers})
        volumes = pd.DataFrame({ticker: price_data[ticker]['Volume'] for ticker in tickers})
    else:
        close_prices = pd.DataFrame({tickers[0]: price_data['Close']})
        volumes = pd.DataFrame({tickers[0]: price_data['Volume']})
    
    fundamentals = {}
    for ticker in tickers:
        try:
            print(f"Fetching fundamentals for {ticker}...")
            stock = yf.Ticker(ticker)
            info = stock.info
            
            fundamentals[ticker] = {
                'marketCap': info.get('marketCap', np.nan),
                'trailingPE': info.get('trailingPE', np.nan),
                'category': ticker_category[ticker]
            }
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            fundamentals[ticker] = {
                'marketCap': np.nan,
                'trailingPE': np.nan,
                'category': ticker_category[ticker]
            }
    
    fundamental_df = pd.DataFrame.from_dict(fundamentals, orient='index')
    
    return close_prices, volumes, fundamental_df

def calculate_features(prices_df, volumes_df):
    daily_returns = prices_df.pct_change().dropna()
    
    features = pd.DataFrame(index=daily_returns.index)
    
    for ticker in prices_df.columns:
        # 1-week return
        features[f'{ticker}_return_1w'] = prices_df[ticker].pct_change(5)
        
        # 3-month return
        features[f'{ticker}_return_3m'] = prices_df[ticker].pct_change(63)
        
        # 1-year return
        features[f'{ticker}_return_1y'] = prices_df[ticker].pct_change(252)
        
        features[f'{ticker}_volume_20d'] = volumes_df[ticker].rolling(window=20).mean()
    
    features = features.dropna()
    
    return daily_returns, features


# get info from yahoo finance of Features
#  include historical price change, historical volume change, market capitalization,
#  price to earnings ratio, and sector classifications.

# put in all one dataframe
# test to see nan values